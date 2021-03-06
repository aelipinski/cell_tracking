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
from datetime import datetime

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
    export_df = export_df.drop(columns=['index','TRACK_INDEX'])
    # Converts to csv for final export 
    return export_df.to_csv(index=False)

def combine_data(input_data):
    df_list = []
    for csv in input_data:
        df_list.append(pd.read_csv(csv))
    return pd.concat(df_list)

@st.experimental_memo(suppress_st_warning=True,show_spinner=True)
def boschloos_test(bosch_vid1,bosch_vid2,bosch_group1,bosch_group2):
    df = st.session_state['input_df']
    df = df[df['meta_vidID'].isin([bosch_vid1,bosch_vid2])]
    df = df[df['GROUP_NAME'].isin([bosch_group1,bosch_group2])]
    contingency_table  = pd.crosstab(df['GROUP_NAME'],df['meta_vidID'])
    exact = stats.boschloo_exact(contingency_table, alternative = 'two-sided')
    alpha = 0.05
    if exact.pvalue > alpha:
        pval_string = 'The two groups are **NOT** significantly different [P-value = {}, Alpha = 0.05]'.format(round(exact.pvalue,2))
    else:
        pval_string = 'The two groups are significantly different [P-value = {}, Alpha = 0.05]'.format(round(exact.pvalue,2))
    # st.session_state['bosch_results'] = ('*Boschloo 2x2 Test*',contingency_table,pval_string)
    return '**Boschloo 2x2 Test**',contingency_table,pval_string

@st.experimental_memo(suppress_st_warning=True,show_spinner=True)
def mann_whitney(group_cat,mw_group1,mw_group2,dep_var):
    df = st.session_state['input_df'][[group_cat,dep_var]]
    x = df[df[group_cat] == mw_group1][dep_var]
    y = df[df[group_cat] == mw_group2][dep_var]
    _,pvalue = stats.mannwhitneyu(x,y)
    alpha = 0.05
    if pvalue > alpha:
        pval_string = '**Result:** The distributions are **NOT** significantly different between the groups.'
    else:
        pval_string = '**Result** The distributions are significantly different between the groups.'
    # st.session_state['bosch_results'] = ('*Boschloo 2x2 Test*',contingency_table,pval_string)
    group_string = '**Groups:** {} vs. {}'.format(mw_group1,mw_group2)
    feature_string = '**Feature:** {}'.format(dep_var)
    pval_string2 = '[P-value = {}, Alpha = 0.05]'.format(round(pvalue,2))
    return '**Mann-Whitney Test**',group_string,feature_string,pval_string,pval_string2


@st.experimental_memo(suppress_st_warning=True,show_spinner=True)
def selector(df,target,algo,cat_or_cont,remove):
    #Filters dataframe by removing the remove features before running selector
    df = df.drop(columns=remove)
    # if filter_contains:
    #     try:
    #         filter_regex = filter_contains.replace(" ","").replace(",","|")  
    #         df = df[df.columns.drop(list(df.filter(regex=re.compile(filter_regex, re.IGNORECASE))))]
    #     except:
    #         st.write("Filter words must be separated by commas with no spaces or special characters.")
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

# def joint_distribution(focus,include_ungrouped,vid_selection):
#     target = 'GROUP_ID'
#     df = st.session_state['numeric_df']
#     df = df[st.session_state['input_df']['meta_vidID'].isin(vid_selection)]
#     if include_ungrouped == False:
#         df = df[df.GROUP_ID != 0]
#     x = df[focus]
#     y = df[target]
#     zx = np.abs(stats.zscore(x))
#     zy = np.abs(stats.zscore(y))
#     xind = np.where(zx < 10)[0]
#     yind = np.where(zy < 10)[0]
#     indices = list(set(xind) & set(yind) )
#     x = x.iloc[indices]
#     y = y.iloc[indices]
#     plot_dict = {focus:x,target:y}
#     plot_df = pd.DataFrame(plot_dict)
#     fig, ax = plt.subplots(1,3,figsize = (12,4))
#     ax[0].hist(x,bins="scott",color="#F63366")
#     ax[0].set_xlabel(focus)
#     ax[0].set_title("Selected Feature")
#     sns.kdeplot(ax=ax[1],data=plot_df,x=focus, y=target,color="#F63366")
#     ax[1].set_title("Selected vs Target")
#     ax[2].hist(y,bins="scott",color="#F63366",orientation="horizontal")
#     ax[2].set_xlabel(target)
#     ax[2].set_title("Target Feature")
#     fig.tight_layout()
#     return st.pyplot(fig)

# @st.experimental_memo(suppress_st_warning=True,show_spinner=True)
def violin_plots(focus,include_ungrouped, vid_selection, target, violin_legend, remove_outliers):
    df = st.session_state['input_df']
    df = df[st.session_state['input_df']['meta_vidID'].isin(vid_selection)]
    if include_ungrouped == False:
        df = df[df.GROUP_ID != 0]
    fig, ax = plt.subplots(1,1,figsize = (10,6))
    if len(df.meta_vidID.unique()) == 2:
        split_val = True
    else:
        split_val = False
    if violin_legend:
        if target == 'GROUP_NAME':
            hue_val = "meta_vidID"
        else:
            hue_val = 'GROUP_NAME'
    else:
        hue_val = None
    if remove_outliers:
        z = np.abs(stats.zscore(df[focus]))
        z_ind = np.where(z < 2)[0]
        df = df.iloc[z_ind]
    ax = sns.violinplot(x=target, y=focus, hue=hue_val, data=df, \
        palette='muted', split=split_val, scale="count", inner='box')
    fig.tight_layout()
    return st.pyplot(fig)

def make_stacked_bars():
    df = st.session_state['input_df']
    agg_df = df.groupby(['meta_vidID','GROUP_NAME']).count()['LABEL'].reset_index()
    agg_df.columns = ['File','Group','Count']
    agg_df = agg_df.pivot(index="File", columns="Group", values="Count").fillna(0)
    agg_df = agg_df.drop(columns=['Ungrouped'])
    fig, ax = plt.subplots(1,1,figsize = (10,4))
    agg_df.astype(int).plot.barh(stacked=True,ax=ax)
    fig.tight_layout()
    return fig

def get_metadata_summary():
    df = st.session_state['input_df']
    regex_meta = re.compile('meta', re.IGNORECASE)
    meta_cols = [i for i in df.columns if regex_meta.match(i)]
    df = df[meta_cols].drop_duplicates()
    df = df.set_index('meta_vidID')
    df.columns = [' '.join(i.split('_')[1:]).capitalize() for i in df.columns]
    return df 

def initialize_session_state_label(spots_data, track_data, img_height, scale_factor):
    # Used to reset all the session states values upon start-up or when the 'reset' button is clicked on 'Label' page
    st.session_state['count'] = 1
    st.session_state['groups'] = dict()
    st.session_state['group_stats'] = pd.DataFrame(columns=["Group Name", "Number of Tracks"])
    st.session_state['selections'] = dict.fromkeys(['im','draw'])
    st.session_state.selections['im'] = Image.new('RGB', (800, int(img_height/scale_factor)), (240, 242, 246))
    st.session_state.selections['draw'] = ImageDraw.Draw(st.session_state.selections['im'])
    st.session_state['spots_df'], st.session_state['spots_array'] = process_spots_data(spots_data)
    st.session_state['track_df'] = process_track_data(track_data)   
    st.session_state.group_stats.loc[0] = ['Ungrouped', sum(pd.isna(st.session_state.track_df['GROUP_ID']))]
    st.session_state['output_options'] = ['All Groups','Ungrouped']
    st.session_state['calib_angle'] = 0.0
    st.session_state['default_vid_id'] = 'vid_' + datetime.today().strftime('%y%m%d_%H%M%S')
    st.experimental_rerun()

def initialize_session_state_analysis(input_data):
    st.session_state['input_df'] = combine_data(input_data)
    st.session_state['numeric_df'] = st.session_state['input_df'].select_dtypes(['number'])
    st.session_state['features_list'] = st.session_state['numeric_df'].columns
    regex_meta = re.compile('meta', re.IGNORECASE)
    st.session_state['features_no_meta'] = [i for i in st.session_state['features_list'] if not regex_meta.match(i)]
    st.session_state['stacked_bars'] = make_stacked_bars()
    st.session_state['metadata'] = get_metadata_summary()
    st.session_state['vid_list'] = st.session_state['input_df']['meta_vidID'].unique()
    pairs = st.session_state['input_df'][['GROUP_ID','GROUP_NAME']].drop_duplicates()
    pairs = pairs[pairs.GROUP_ID != 0]
    st.session_state['groups_dict'] = dict(list(zip(pairs['GROUP_ID'],pairs['GROUP_NAME'])))
    # regex_importance = re.compile('GROUP_ID|track|number|.*location|.*displacement|.*position|.*frame|.*calib|longest|.*distance|.*speed|confinement|.*linearity|.*change', re.IGNORECASE)
    # st.session_state['features_importance'] = [i for i in st.session_state['features_list'] if not regex_importance.match(i)]
    st.session_state['features_importance'] = [
        "meta_true_angle",
        "meta_ridge_number",
        "meta_ridge_spacing",
        "meta_ridge_width",
        "meta_gap_size",
        "meta_gutter_number",
        "meta_gutter_size",
        "meta_channel_width",
        "meta_flowrate",
        "meta_sheath_number",
        "meta_cell_conc",
        "spots_AREA_mean",
        "spots_MEAN_INTENSITY_CH1_mean"
    ]
    st.session_state['importance_df'] = st.session_state['numeric_df'][st.session_state['features_importance']+['GROUP_ID']]
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
            vid_id = st.text_input("Video ID",value=st.session_state['default_vid_id'])
            vid_date = st.date_input("Date", datetime.today())
            frame_rate = st.number_input("Frame Rate (fps)",value=3000)
            flow_direction = st.selectbox("Flow Direction",['Left','Right'])
            ridge_angle = st.number_input("Ridge Angle (deg)",value=15.0)

            # Calculate and display calibrated angle and true angle 
            st.write("Calibration angle:",round(st.session_state.calib_angle,2))
            true_angle = st.session_state.calib_angle + ridge_angle
            st.write("True angle:",round(true_angle,2))

            ridge_number = st.number_input("Ridge Number",value=5,min_value=1, max_value=10, step=1)
            ridge_spacing = st.number_input("Ridge Spacing (um)",value=240)
            ridge_width = st.number_input("Ridge Width (um)",value=20)
            ridge_design = st.selectbox("Ridge Design",['Straight','Chevron'])
            gap_size = st.number_input("Gap Size (um)",value=5.0)
            gutter_number = st.number_input("Gutter Number",value=2,min_value=0, max_value=2, step=1)
            gutter_size = st.number_input("Gutter Size (um)",value=80)
            channel_width = st.number_input("Channel Width (um)",value=560)
            flowrate = st.number_input("Fluid Flowrate (uL/min)",value=25)
            sheath_number = st.number_input("Sheath Number",value=2,min_value=0, max_value=2, step=1)
            media = st.selectbox("Media", ['Media', 'Flow Buffer', 'Other'],index=1)	
            cell_conc = st.number_input('Cell Concentration (million cells/mL)',value=1.0)

            metadata_dict = {
                "meta_vidID":vid_id,
                "meta_date":vid_date,
                "meta_flow_direction":flow_direction,
                "meta_frame_rate":frame_rate,
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
            draw_mode = st.selectbox("Mode",("Labeling","Calibration"))
            output_groups = st.selectbox("Select Group to Display",st.session_state.output_options) 
            if draw_mode == "Labeling":
                label_type = st.radio("Label Type", ("Include","Exclude"))  
                default_group_name = "group_"+str(st.session_state['count'])
                group_name = st.text_input("Label Name", value= default_group_name, help="Add name for group")      

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
            key="main"
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
                st.session_state.groups[group_name] = {'tracks':track_set}

                # Add output drawing for current group to the session state with a random color 
                # Draws all points for the tracks belonging to the group s
                draw_points = st.session_state.spots_df[st.session_state.spots_df['TRACK_ID'].isin(track_set)]
                coords = tuple(zip(draw_points.POSITION_X.astype(float)/scale_factor,draw_points.POSITION_Y.astype(float)/scale_factor))
                st.session_state.groups[group_name]['points'] = coords
                color = output_colors[(group_id-1) % len(output_colors)]
                st.session_state.groups[group_name]['color'] = color
                st.session_state.selections['draw'].point(coords, fill=color)
                st.session_state.group_stats.loc[group_id] = [group_name, len(track_set)]
                st.session_state.track_df.loc[st.session_state.track_df['TRACK_ID'].isin(track_set),['GROUP_ID','GROUP_NAME']] = [group_id, group_name]
                st.session_state.group_stats.loc[0] = ['Ungrouped', sum(pd.isna(st.session_state.track_df['GROUP_ID']))]
                st.session_state.output_options.append(group_name)

                # Increment the group ID
                st.session_state.count += 1
                st.experimental_rerun()
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

        st.write('### Group Summary')
        st.dataframe(st.session_state.group_stats)

# ----------------------------------- PAGE 2: Analysis -----------------------------------

def analysis_page():
    # Add CSV uploaders within expander 
    with st.sidebar.expander('Load Data'):
        if st.checkbox('Demo Mode',on_change=clear_session_state):
            input_data = [
                'demo/bsa.csv',
                'demo/20_10_20.csv',
                'demo/30_15_30.csv',
                'demo/10_5_10.csv'
            ]
        else:
            input_data = st.file_uploader("Upload Labeled Tracking Data",type='csv',accept_multiple_files=True, \
                help="Data must be a CSV file exported from the 'Labeling and Annotation page'.")

    if input_data:
        # Initialize session state variables 
        if 'input_df' not in st.session_state:
            initialize_session_state_analysis(input_data)
        
        with st.expander('Data Summary',expanded=True):
            st.pyplot(st.session_state['stacked_bars'])
            st.dataframe(st.session_state['metadata'])

        with st.sidebar.expander('Feature Importance'):

            # cat_or_cont = st.selectbox("Target Data Type",("Categorical", "Continuous"),index=0,\
            #     help = "This will help determine which type of algorithm to run - classification or regression.")

            # max_num = int(len(st.session_state['features_list'])-1)
            num_features = st.number_input("Number of Features to Select",min_value=1,max_value=10,value=3,\
                help="Consider the purpose of this feature selection. When in doubt, select a higher number of features.")

            remove = st.multiselect("Remove Specific Features", st.session_state['features_importance'],\
                help = "Remove certain features from the chosen features list.")
            
            # filter_contains = st.text_input("Remove Features Containing")

            # algo = st.selectbox("Choose feature selection algorithm",["MRMR","Mutual Info"],\
            #     help = "MRMR: good for redundant data, Mutual Info: good for non-linear data with few redundant variables or when redundancy doesn't matter.")

        with st.expander("Feature Importance",expanded=True):
            chosen = selector(st.session_state['importance_df'],'GROUP_ID','Mutal Info','Categorical',remove)
            st.write(chosen[:int(num_features)])

        group_cat_dict = {'File':'meta_vidID','Group':'GROUP_NAME'}

        visual_expander = st.sidebar.expander('Distribution Plots')
        vid_selection_violin = visual_expander.multiselect("Included Files",st.session_state['vid_list'],st.session_state['vid_list'],key='violin')
        # show_joint = visual_expander.checkbox('Show Joint Distribution',True)
        # joint_focus = visual_expander.selectbox('Joint Distribution Feature',st.session_state['features_list']) 
        # show_violin = visual_expander.checkbox('Show Violin Plots',True)
        violin_focus = visual_expander.selectbox('Violin Plot Feature',st.session_state['features_no_meta'],index=27) 
        primary_group = visual_expander.selectbox('Primary Category',['File','Group'],index=1,key='violin_primary')
        violin_legend = visual_expander.checkbox('Split By Secondary Category', True)
        remove_outliers = visual_expander.checkbox('Remove Outliers', True)
        include_ungrouped = visual_expander.checkbox('Include Ungrouped Tracks',False)
        with st.expander('Distribution Plots',expanded=True):
            # if show_joint:
            #     joint_distribution(joint_focus,include_ungrouped,vid_selection)
            # if show_violin:
            violin_plots(violin_focus,include_ungrouped,vid_selection_violin,group_cat_dict[primary_group],violin_legend,remove_outliers)
        
        stat_expander = st.sidebar.expander('Statistical Test')
        stat_test = stat_expander.selectbox("Test Type",['Boschloo','Mann-Whitney'],index=1)
        if stat_test == 'Boschloo':
            bosch_vid1 = stat_expander.selectbox('File #1',st.session_state['vid_list'],index=0)
            bosch_vid2 = stat_expander.selectbox('File #2',st.session_state['vid_list'],index=1)
            bosch_group1 = stat_expander.selectbox('Group #1',st.session_state['groups_dict'].values(),index=0)
            bosch_group2 = stat_expander.selectbox('Group #2',st.session_state['groups_dict'].values(),index=1)
        if stat_test == 'Mann-Whitney':
            dep_var = stat_expander.selectbox('Comparison Feature',st.session_state['features_no_meta'],index=27)
            group_cat = stat_expander.selectbox('Category',['File','Group'],index=1)
            if group_cat == 'File':
                mw_group1 = stat_expander.selectbox('File #1',st.session_state['vid_list'],index=0)
                mw_group2 = stat_expander.selectbox('File #2',st.session_state['vid_list'],index=1)
            else:
                mw_group1 = stat_expander.selectbox('Group #1',st.session_state['groups_dict'].values(),index=0)
                mw_group2 = stat_expander.selectbox('Group #2',st.session_state['groups_dict'].values(),index=1)

        with st.expander("Statistical Test", expanded = True):
            if stat_test == 'Boschloo':
                results = boschloos_test(bosch_vid1,bosch_vid2,bosch_group1,bosch_group2)
                for result in results:
                    st.write(result)
            if stat_test == 'Mann-Whitney':
                results = mann_whitney(group_cat_dict[group_cat],mw_group1,mw_group2,dep_var)
                for result in results:
                    st.write(result)
        # if stat_expander.button('Run Test'):
        #     if stat_test == 'Boschloo (2 groups)':
        #         boschloos_test(bosch_vid1,bosch_vid2,bosch_group1,bosch_group2)
        # if 'bosch_results' in st.session_state:
        #     with st.expander("Statistical Test",expanded=True):
        #         for result in st.session_state['bosch_results']:
        #             st.write(result)
        

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
# Improve group summary (add columns)
# allow individual group deletion 
# add subtitle/description to page explaining purpose 
# Automate meta dictionary creation 
# Add calibrate button confirmation (show calib angle after click)
# Make output a session state image with an update variable (add if to show all or individual, updates individual) ***
# Make scale factor and img height session state variables 
# Round metadata files to 2 decimals before export

# ---- PART 2 ----
# add subtitle/description to page explaining purpose 
# handle case when # of group names is not exactly 2 (select 2 groups if there's more than 2) or 2-way ANOVA 
# Logistic Regression to predict group membership based on cell + device attributes 
# Kruskall Wallis for ordinal groups 
# Remove nonsense fields (ID, index, etc)
# Add optional button to remove geometric fields from feature importance (On by default) 
# Fix Feature Importance variables *** 



# ---- GENERAL ----
# Add help popup for all streamlit widgets 
# write manual