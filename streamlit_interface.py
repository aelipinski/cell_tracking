from curses import meta
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st
st.set_page_config(layout="wide")
from streamlit_drawable_canvas import st_canvas
import numpy as np
from feature_select import FeatureSelector
import re
import scipy.stats as stats
import matplotlib.pyplot as plt 
import seaborn as sns

# ----------------------------------- FUNCTIONS -----------------------------------

# Functions to clean up TrackMate CSVs and convert to dataframes
@st.experimental_memo
def process_spots_data(spots_data):
    spots_df = pd.read_csv(spots_data)
    spots_df = spots_df.drop([0,1,2])
    spots_df = spots_df.reset_index()
    spots_df['TRACK_ID'] = spots_df['TRACK_ID'].astype(int)
    spots_array = np.array(spots_df[["POSITION_X", "POSITION_Y"]]).astype(float)
    return spots_df, spots_array

@st.experimental_memo
def process_track_data(track_data):
    track_df = pd.read_csv(track_data)
    track_df = track_df.drop([0,1,2])
    track_df = track_df.reset_index()
    track_df['TRACK_ID'] = track_df['TRACK_ID'].astype(int)
    track_df[['GROUP_ID','GROUP_NAME']] = None
    return track_df

# Finds line coefficients for each polygon edge
def line_solver(x0, y0, x1, y1, scale_factor=1):
    a = scale_factor * float(y1 - y0)
    b = scale_factor * float(x0 - x1)
    c = scale_factor * (- a*x0 - b*y0)
    return np.array([a,b,c])

# Finds all line coeffcients for all edges in all polygons 
def get_edge_coefs(canvas_result, scale_factor):
    canvas_objects = canvas_result.json_data["objects"]
    num_poly = len(canvas_objects)
    poly_array = {}
    for i in range(num_poly):
        poly = canvas_result.json_data["objects"][i]["path"]
        num_vert = len(poly) - 1 
        coef_mat = np.zeros((3,num_vert))
        for edge in range(num_vert):
            coef_mat[:,edge] = line_solver(poly[edge][1], poly[edge][2], poly[(edge + 1) % num_vert][1], poly[(edge + 1) % num_vert][2], scale_factor)
        poly_array[i] = coef_mat
    return poly_array

# Checks all spots and determines if they fall inside a given polygon (must be a convex polygon!)
def test_all_points(spots_array, all_coefs):
    prod_mat = spots_array @ all_coefs[:2,:] + all_coefs[2,:] 
    sideA = prod_mat<=0
    sideB = ~sideA
    point_indices = np.all(sideA, axis=1) + np.all(sideB, axis=1)
    return point_indices

# Calculates angle from line endpoints
def get_angle(x1,x2,y1,y2):
    angle = np.degrees(np.arctan((y2-y1)/(x2-x1)))
    return angle

def clear_session_state():
    for key in st.session_state.keys():
        del st.session_state[key]

def prepare_export(metadata_dict):
    # Dictionary specifying which columns to keep and which aggregations to perform from spots.csv
    spots_agg_dict = {
        'AREA':'mean',
        'POSITION_X':['min','mean','max'],
        'POSITION_Y':['min','mean','max'],
        'MEAN_INTENSITY_CH1':'mean'
    }
    # Creates a subset of spots_df using only 'TRACK_ID' and the chosen columns to be grouped/aggregated
    spots_agg_df = st.session_state['spots_df'][['TRACK_ID']+list(spots_agg_dict.keys())].astype(float)
    # Groups by 'TRACK_ID' and performs corresponding aggregations according to spots_agg_dict
    spots_agg_df = spots_agg_df.groupby('TRACK_ID').agg(spots_agg_dict)
    # Improves columns names by adding prefix and combining multi-index column names
    spots_colnames = pd.Index(['spots_' + e[0] + '_' + e[1] for e in spots_agg_df.columns.tolist()])
    spots_agg_df.columns = spots_colnames
    # Adds metadata to track_df as repeating/fixed value for all rows
    for key in metadata_dict.keys():
        st.session_state['track_df'][key] = metadata_dict[key]
    # Merges track_df and the grouped spots_df using 'TRACK_ID' as key
    export_df = pd.merge(st.session_state['track_df'],spots_agg_df,on='TRACK_ID')
    export_df.loc[export_df['GROUP_ID'].isna(),['GROUP_ID','GROUP_NAME']] = [0,"Ungrouped"]
    # Converts to csv for final export 
    return export_df.to_csv(index=False)

def combine_data(input_data):
    df_list = []
    for csv in input_data:
        df_list.append(pd.read_csv(csv))
    return pd.concat(df_list)

def boschloos_test():
    df = st.session_state['input_df']
    df = df[df.GROUP_ID != 0]
    contingency_table  = pd.crosstab(df['GROUP_NAME'],df['meta_vidID'])
    exact = stats.boschloo_exact(contingency_table, alternative = 'two-sided')
    alpha = 0.05
    if exact.pvalue > alpha:
        pval_string = 'The two groups are **NOT** significantly different [P-value = {}, Alpha = 0.05]'.format(round(exact.pvalue,2))
    else:
        pval_string = 'The two groups are significantly different [P-value = {}, Alpha = 0.05]'.format(round(exact.pvalue,2))
    # st.session_state['stat_results'] = (contingency_table,exact.pvalue)
    st.session_state['stat_results'] = (contingency_table,pval_string)

@st.experimental_memo(suppress_st_warning=True,show_spinner=True)
def selector(df,target,algo,cat_or_cont,remove,filter_contains):
    #Filters dataframe by removing the remove features before running selector
    df = df.drop(columns=remove)
    if filter_contains:
        try:
            filter_regex = filter_contains.replace(" ","").replace(",","|")  
            df = df[df.columns.drop(list(df.filter(regex=re.compile(filter_regex, re.IGNORECASE))))]
        except:
            st.write("Filter words must be separated by commas with no spaces or special characters.")
    fs = FeatureSelector(df = df, target_col = target)
    if algo == "MRMR":
        rank = fs.mrmr()
    elif cat_or_cont == "Continuous":
        rank = fs.mutual_info_regress()
        rank = rank['Feature'].tolist()
    else:
        rank = fs.mutual_info_class()
        rank = rank['Feature'].tolist()
    # return pd.DataFrame(rank,columns=['Feature'])
    return rank

def joint_distribution(target,focus,include_ungrouped):
    df = st.session_state['numeric_df']
    if include_ungrouped == False:
        df = df[df.GROUP_ID != 0]
    x = df[focus]
    y = df[target]
    zx = np.abs(stats.zscore(x))
    zy = np.abs(stats.zscore(y))
    xind = np.where(zx < 3)[0]
    yind = np.where(zy < 3)[0]
    indices = list(set(xind) & set(yind) )
    x = x.iloc[indices]
    y = y.iloc[indices]
    plot_dict = {focus:x,target:y}
    plot_df = pd.DataFrame(plot_dict)
    fig, ax = plt.subplots(1,3,figsize = (12,4))
    ax[0].hist(x,bins="scott",color="#F63366")
    ax[0].set_xlabel(focus)
    ax[0].set_title("Selected Feature")
    sns.kdeplot(ax=ax[1],data=plot_df,x=focus, y=target,color="#F63366")
    ax[1].set_title("Selected vs Target")
    ax[2].hist(y,bins="scott",color="#F63366",orientation="horizontal")
    ax[2].set_xlabel(target)
    ax[2].set_title("Target Feature")
    fig.tight_layout()
    return st.pyplot(fig)

def violin_plots(target,focus,include_ungrouped):
    target='GROUP_NAME'
    df = st.session_state['input_df']
    if include_ungrouped == False:
        df = df[df.GROUP_ID != 0]
    fig, ax = plt.subplots(1,1,figsize = (12,4))
    if len(df.meta_vidID.unique()) == 2:
        split_val = True
    else:
        split_val = False
    ax = sns.violinplot(x=target, y=focus, hue="meta_vidID", data=df, \
        palette='muted', split=split_val, scale="count", inner=None)
    fig.tight_layout()
    return st.pyplot(fig)

def initialize_session_state_label(spots_data, track_data, img_height, scale_factor):
    # Used to reset all the session states values upon start-up or when the 'reset' button is clicked on 'Label' page
    st.session_state['count'] = 1
    st.session_state['groups'] = dict()
    st.session_state['group_stats'] = pd.DataFrame(columns=["Group ID", "Group Name", "Number of Tracks"])
    st.session_state['selections'] = dict.fromkeys(['im','draw'])
    st.session_state.selections['im'] = Image.new('RGB', (800, int(img_height/scale_factor)), (240, 242, 246))
    st.session_state.selections['draw'] = ImageDraw.Draw(st.session_state.selections['im'])
    st.session_state['spots_df'], st.session_state['spots_array'] = process_spots_data(spots_data)
    st.session_state['track_df'] = process_track_data(track_data)   
    st.session_state['output_options'] = ['All Groups','Ungrouped']
    st.session_state['calib_angle'] = 0.0
    st.experimental_rerun()

def initialize_session_state_analysis(input_data):
    st.session_state['input_df'] = combine_data(input_data)
    st.session_state['numeric_df'] = st.session_state['input_df'].select_dtypes(['number'])
    st.session_state['features_list'] = st.session_state['numeric_df'].columns
    regex_meta = re.compile('meta', re.IGNORECASE)
    st.session_state['features_no_meta'] = [i for i in st.session_state['features_list'] if not regex_meta.match(i)]
    st.experimental_rerun()

# ----------------------------------- PAGE 1: Labeling  -----------------------------------

def label_page():

    # Add CSV uploaders for spots and tracking data and image uploader to the sidebar within expander 
    with st.sidebar.expander('Load Data'):
        if st.checkbox('Demo Mode',on_change=clear_session_state):
            spots_data = "demo/spots.csv"
            track_data = "demo/tracks.csv"
            bg_image = "demo/bg1.png"
        else:
            spots_data = st.file_uploader("Spots Data from TrackMate",type='csv',accept_multiple_files=False, \
                help="Data must be a CSV file exported from TrackMate for the 'Spots'.")

            track_data = st.file_uploader("Track Data from Trackmate ",type='csv',accept_multiple_files=False, \
                help="Data must be a CSV file exported from TrackMate for the 'Tracks'.")    

            bg_image = st.file_uploader("Background Composite Image:", type=["png", "jpg"], accept_multiple_files=False, \
                help="Data should be an image from TrackMate showing all tracks overlapping a background video frame.")  

    # Dictionary to set fill and stroke colors for include and exlude polygon objects 
    poly_color = {"Include":["rgba(10, 255, 0, 0.3)","rgba(10, 255, 0, 1)"],\
                "Exclude":["rgba(255, 10, 0, 0.3)","rgba(255, 10, 0, 1)"],\
                "Calibration":["rgba(0, 0, 0, 0)","rgba(133, 229, 255, 1)"]}

    poly_type = {"Labeling":"polygon","Calibration":"line"}

    output_colors = [(228,26,28),(55,126,184),(77,175,74),(152,78,163), \
        (255,127,0),(255,255,51),(166,86,40),(247,129,191),(153,153,153)]

    # Draws Input and Output images if a background image has been loaded 
    if spots_data and track_data and bg_image:

        # Read image dimensions and determine scaling factor for a width of 800 
        img_width,img_height = Image.open(bg_image).size
        scale_factor = img_width/800

        # Initialize session state variables for iterative group creation
        # This prevents the groups from being deleted with each Streamlit run 
        if 'count' not in st.session_state:
            initialize_session_state_label(spots_data, track_data, img_height, scale_factor)

        st.write('### Input')

        # Specify metadata that applies to entire video 
        with st.sidebar.expander('Video Metadata'):
            vid_id = st.text_input("Video ID")
            ridge_angle = st.number_input("Ridge Angle (deg)")

            # Calculate and display calibrated angle and true angle 
            st.write("Calibration angle:",round(st.session_state.calib_angle,2))
            true_angle = st.session_state.calib_angle + ridge_angle
            st.write("True angle:",round(true_angle,2))

            ridge_number = st.number_input("Ridge Number",min_value=1, max_value=10, step=1)
            ridge_spacing = st.number_input("Ridge Spacing (um)")
            ridge_width = st.number_input("Ridge Width (um)")
            ridge_design = st.selectbox("Ridge Design",['Straight','Chevron'])
            gap_size = st.number_input("Gap Size (um)")
            gutter_number = st.selectbox("Gutter Number",[0,1,2])
            gutter_size = st.number_input("Gutter Size (um)")
            channel_width = st.number_input("Channel Width (um)")
            flowrate = st.number_input("Fluid Flowrate (uL/min)")
            sheath_number = st.selectbox("Sheath Number",[0,1,2])
            media = st.selectbox("Media", ['Media', 'Flow Buffer', 'Other'])	
            cell_conc = st.number_input('Cell Concentration (million cells/mL)')

            metadata_dict = {
                "meta_vidID":vid_id,
                "meta_ridge_angle":ridge_angle,
                "meta_calib_angle":st.session_state.calib_angle,
                "meta_true_angle":true_angle,
                "meta_ridge_number":ridge_number,
                "meta_ridge_spacing":ridge_spacing,
                "meta_ridge_width":ridge_width,
                "meta_ridge_design":ridge_design,
                "meta_gap_size":gap_size,
                "meta_gutter_number":gutter_number,
                "meta_gutter_size":gutter_size,
                "meta_channel_width":channel_width,
                "meta_flowrate":flowrate,
                "meta_sheath_number":sheath_number,
                "meta_media":media,
                "meta_cell_conc":cell_conc
            }
            
        # Specify canvas parameters in application
        with st.sidebar.expander("Drawing Options"):
            draw_mode = st.radio("Mode",("Labeling","Calibration"))
            if draw_mode == "Labeling":
                label_type = st.radio("Label Type", ("Include","Exclude"))  
                group_name = st.text_input("Label Name", help="Add name for group")            

        # Create drawing canvas using API
        canvas_result = st_canvas(
            fill_color=poly_color[label_type][0] if draw_mode == "Labeling" else poly_color["Calibration"][0],  
            stroke_width=1,
            stroke_color=poly_color[label_type][1] if draw_mode == "Labeling" else poly_color["Calibration"][1],
            background_color= "#eee",
            background_image=Image.open(bg_image),
            update_streamlit=True,
            height=img_height/scale_factor,
            width = 800,
            drawing_mode=poly_type[draw_mode],
            key=draw_mode
        )

        if draw_mode == "Labeling":
            # Creates new group when button is clicked (checks points, add labels to corresponding tracks, draws output)
            if st.button('Create Group') and len(canvas_result.json_data["objects"]) > 0:

                # Determine the number of polygons to check and get edge coefficients for each polygon 
                num_poly = len(canvas_result.json_data["objects"])
                edge_coefs = get_edge_coefs(canvas_result, scale_factor)

                # Initialize positive and negative track sets 
                track_set = set()
                neg_set = set()

                # Iterate through all polygons and keep track of track IDs to add or subtract 
                for poly in range(num_poly):

                    # returns index values of points included within polygon
                    point_indices = test_all_points(st.session_state.spots_array,edge_coefs[poly])

                    # retreive corresponding tracks for the bounded points
                    poly_set = set(st.session_state.spots_df["TRACK_ID"][point_indices])

                    # Check if polygon is an include or exclude type by looking at it's color (red or green)
                    # Then add or remove track IDs depending on type

                    if canvas_result.json_data["objects"][poly]["fill"] == poly_color["Include"][0]:
                        track_set = track_set | poly_set
                    else:
                        neg_set = neg_set | poly_set
                
                # Create final track set by subtracting the negative set 
                track_set = track_set - neg_set

                # Aadd the track set and groupd name to the session state 
                group_id = st.session_state.count
                st.session_state.groups[group_id] = {'name':group_name, 'tracks':track_set}

                # Add output drawing for current group to the session state with a random color 
                # Draws all points for the tracks belonging to the group s
                draw_points = st.session_state.spots_df[st.session_state.spots_df['TRACK_ID'].isin(track_set)]
                coords = tuple(zip(draw_points.POSITION_X.astype(float)/scale_factor,draw_points.POSITION_Y.astype(float)/scale_factor))
                st.session_state.groups[group_id]['points'] = coords
                color = output_colors[(group_id-1) % len(output_colors)]
                st.session_state.groups[group_id]['color'] = color
                st.session_state.selections['draw'].point(coords, fill=color)
                st.session_state.group_stats.loc[group_id-1] = [group_id, group_name, len(track_set)]
                st.session_state.track_df.loc[st.session_state.track_df['TRACK_ID'].isin(track_set),['GROUP_ID','GROUP_NAME']] = [group_id, group_name]
                st.session_state.group_stats.loc[len(st.session_state.group_stats)] = [len(st.session_state.group_stats)+1, 'Ungrouped', sum(pd.isna(st.session_state.track_df['GROUP_ID']))]
                st.session_state.output_options.append(group_id)

                # Increment the group ID
                st.session_state.count += 1
        else:
            if st.button('Calibrate Angle'):
                # Only creates calibration angle if a single line is drawn
                if len(canvas_result.json_data["objects"])==1:
                    x1 = canvas_result.json_data["objects"][0]["x1"]
                    x2 = canvas_result.json_data["objects"][0]["x2"]
                    y1 = canvas_result.json_data["objects"][0]["y1"]
                    y2 = canvas_result.json_data["objects"][0]["y2"]
                    # Checks which way the line is facing (changes based on direction of creation by user)
                    # Necessary to ensure consistent sign of angle, regardless of drawing direction (+ or -)
                    if x1 < x2:
                        st.session_state['calib_angle'] = get_angle(x1,x2,y1,y2)
                    else:
                        st.session_state['calib_angle'] = get_angle(x2,x1,y2,y1)
                    st.experimental_rerun()
                # If zero or multiple lines are drawn, warns user to correct it
                else:
                    st.write("Please create exactly 1 line.")


        with st.sidebar.expander("Display Settings"):
            show_stats = st.checkbox('Show Group Summary',False)
            output_groups = st.selectbox("Select Group to Display",st.session_state.output_options)

        # Add Export and Reset Buttons side by side
        col1, col2 = st.sidebar.columns([1,2])

        with col1:
            st.download_button(
                label="Export",
                data=prepare_export(metadata_dict),
                file_name=vid_id+'.csv' if vid_id else 'export.csv',
                mime='text/csv'
            )
        with col2:
            if st.button('Reset'):
                initialize_session_state_label(spots_data, track_data, img_height, scale_factor)

        # Draw Output if background image is loaded 
        st.write('### Output')

        if output_groups == 'All Groups':
            st.image(st.session_state.selections['im'])
        elif output_groups == 'Ungrouped':
            if len(st.session_state['groups']) > 0:
                untracked = st.session_state.track_df.loc[pd.isna(st.session_state.track_df['GROUP_ID']),'TRACK_ID'].unique()
                draw_points = st.session_state.spots_df[st.session_state.spots_df['TRACK_ID'].isin(untracked)]
            else:
                draw_points = st.session_state.spots_df
            coords = tuple(zip(draw_points.POSITION_X.astype(float)/scale_factor,draw_points.POSITION_Y.astype(float)/scale_factor))
            im_single = Image.new('RGB', (800, int(img_height/scale_factor)), (240, 242, 246))
            draw_single = ImageDraw.Draw(im_single)
            draw_single.point(coords, fill = (0,0,0))
            st.image(im_single)
        else:
            im_single = Image.new('RGB', (800, int(img_height/scale_factor)), (240, 242, 246))
            draw_single = ImageDraw.Draw(im_single)
            draw_single.point(st.session_state.groups[output_groups]['points'], fill = st.session_state.groups[output_groups]['color'])
            st.image(im_single)

        if len(st.session_state['groups']) > 0 and show_stats:
            st.write('### Group Summary')
            st.dataframe(st.session_state.group_stats)

# ----------------------------------- PAGE 2: Analysis -----------------------------------

def analysis_page():
    # Add CSV uploaders within expander 
    with st.sidebar.expander('Load Data'):
        if st.checkbox('Demo Mode',on_change=clear_session_state):
            input_data = ["demo/demo1.csv","demo/demo2.csv"]
        else:
            input_data = st.file_uploader("Upload Labeled Tracking Data",type='csv',accept_multiple_files=True, \
                help="Data must be a CSV file exported from the 'Labeling and Annotation page'.")

    if input_data:
        # Initialize session state variables 
        if 'input_df' not in st.session_state:
            initialize_session_state_analysis(input_data)

        with st.sidebar.expander('Feature Importance'):

            target = st.selectbox("Target Feature",['GROUP_NAME','GROUP_ID'],index=1, \
                help = "Features will be selected with the ultimate goal of estimating this target feature.")

            cat_or_cont = st.selectbox("Target Data Type",("Categorical", "Continuous"),index=0,\
                help = "This will help determine which type of algorithm to run - classification or regression.")

            # max_num = int(len(st.session_state['features_list'])-1)
            num_features = st.number_input("Number of Features to Select",min_value=1,max_value=10,value=5,\
                help="Consider the purpose of this feature selection. When in doubt, select a higher number of features.")

            remove = st.multiselect("Remove Specific Features", st.session_state['features_list'],\
                help = "Remove certain features from the chosen features list.")
            
            filter_contains = st.text_input("Remove Features Containing")

            algo = st.selectbox("Choose feature selection algorithm",["MRMR","Mutual Info"],\
                help = "MRMR: good for redundant data, Mutual Info: good for non-linear data with few redundant variables or when redundancy doesn't matter.")

        with st.expander("Feature Importance",expanded=True):
            chosen = selector(st.session_state['numeric_df'],target,algo,cat_or_cont,remove,filter_contains)
            st.write(chosen[:int(num_features)])

        stat_expander = st.sidebar.expander('Statistical Tests')
        stat_test = stat_expander.selectbox("Test Type",['Boschloo (2 groups)'])
        if stat_expander.button('Run Test'):
            if stat_test == 'Boschloo (2 groups)':
                boschloos_test()
                
        if 'stat_results' in st.session_state:
            with st.expander("Statistical Tests",expanded=True):
                for result in st.session_state['stat_results']:
                    st.write(result)

        visual_expander = st.sidebar.expander('Exploratory Analysis')
        visual_select = visual_expander.selectbox('Visualization',['Joint Distribution','Violin Plots'])
        include_ungrouped = visual_expander.checkbox('Include Ungrouped Tracks',False)
        if visual_select in ('Joint Distribution','Violin Plots'):
            focus = visual_expander.selectbox('Comparison Feature',st.session_state['features_no_meta']) #features_list
        if visual_expander.button('Display'):
            with st.expander('Exploratory Analysis',expanded=True):
                if visual_select == 'Joint Distribution':
                    joint_distribution(target,focus,include_ungrouped)
                elif visual_select == 'Violin Plots':
                    violin_plots(target,focus,include_ungrouped)
        

# ----------------------------------- Run the app -----------------------------------

# Title and header
st.sidebar.title("TrackMate Analysis")

PAGES = {
    "1. Labeling and Annotation": label_page,
    "2. Analysis": analysis_page
}
page = st.sidebar.selectbox("Page", options=list(PAGES.keys()), on_change=clear_session_state)
PAGES[page]()


# ----------------------------------- NOTES -----------------------------------

# ---- PART 1 ----
# Add convexity checker
# Save pre-made configurations 
# Improve column name formatting (all caps)
# Wrap calculation in functions with st.experimental_memo (leave out markdown parts)
# Improve group summary (add columns, make visible by default)
# allow individual group deletion 
# add subtitle/description to page explaining purpose 
# Provide default name to group name if left blank ***
# add additional metadata fields and default values from Peter ***
# Make ungrouped zero index for summary ***
# Automate meta dictionary creation ***

# ---- PART 2 ----
# Check swithching from demo to regular and back
# add subtitle/description to page explaining purpose 
# handle case when # of group names is not exactly 2 (select 2 videos if there's more than 2) or 2-way ANOVA
# Logistic Regression to predict group membership based on cell + device attributes 
# Kruskall Wallis for ordinal groups
# Add summary table ***
# Add optional button to remove geometric fields from feature importance (On by default) ***
# Figure out GROUP_ID vs GROUP_Name numeric issue ***
# Make plots persistent like stat results and use multi-check option ***
# remove meta fields for violin plots ***
# Add box plots
# Run all combinations of boschloo 
# Add stacked bar graph of groups (both absolute and set to 100% relative)


# ---- GENERAL ----
# write manual