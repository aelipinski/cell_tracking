import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np

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
    return track_df

# Finds line coefficients for each polygon edge
def line_solver(x0, y0, x1, y1):
    a = scale_factor * float(y1 - y0)
    b = scale_factor * float(x0 - x1)
    c = scale_factor * (- a*x0 - b*y0)
    return np.array([a,b,c])

# Finds all line coeffcients for all edges in all polygons 
def get_edge_coefs(canvas_objects):
    num_poly = len(canvas_objects)
    poly_array = {}
    for i in range(num_poly):
        poly = canvas_result.json_data["objects"][i]["path"]
        num_vert = len(poly) - 1 
        coef_mat = np.zeros((3,num_vert))
        for edge in range(num_vert):
            coef_mat[:,edge] = line_solver(poly[edge][1], poly[edge][2], poly[(edge + 1) % num_vert][1], poly[(edge + 1) % num_vert][2])
        poly_array[i] = coef_mat
    return poly_array

# Checks all spots and determines if they fall inside a given polygon (must be a convex polygon!)
def test_all_points(spots_array, all_coefs):
    prod_mat = spots_array @ all_coefs[:2,:] + all_coefs[2,:] 
    sideA = prod_mat<=0
    sideB = ~sideA
    point_indices = np.all(sideA, axis=1) + np.all(sideB, axis=1)
    return point_indices

def initialize_session_state():
    st.session_state['count'] = 1
    st.session_state['groups'] = dict()
    st.session_state['group_stats'] = pd.DataFrame(columns=["Group ID", "Group Name", "Number of Tracks"])
    st.session_state['selections'] = dict.fromkeys(['im','draw'])
    st.session_state.selections['im'] = Image.new('RGB', (800, int(img_height/scale_factor)), (240, 242, 246))
    st.session_state.selections['draw'] = ImageDraw.Draw(st.session_state.selections['im'])
    st.session_state['spots_df'], st.session_state['spots_array'] = process_spots_data(spots_data)
    st.session_state['track_df'] = process_track_data(track_data)   

# ----------------------------------- LAYOUT -----------------------------------

# Title and header
st.sidebar.title("TrackMate Analysis")

# Add CSV uploaders for spots and tracking data and image uploader to the sidebar within expander 
with st.sidebar.expander('Load Data'):
    if st.checkbox('Demo Mode'):
        spots_data = "spots.csv"
        track_data = "tracks.csv"
        bg_image = "bg1.png"
    else:
        spots_data = st.file_uploader("Spots Data from TrackMate",type='csv',accept_multiple_files=False, \
            help="Data must be a CSV file exported from TrackMate for the 'Spots'.")

        track_data = st.file_uploader("Track Data from Trackmate ",type='csv',accept_multiple_files=False, \
            help="Data must be a CSV file exported from TrackMate for the 'Tracks'.")    

        bg_image = st.file_uploader("Background Composite Image:", type=["png", "jpg"], accept_multiple_files=False, \
            help="Data should be an image from TrackMate showing all tracks overlapping a background video frame.")  

# Specify metadata that applies to entire video 
with st.sidebar.expander('Video Metadata'):
    ridge_spacing = st.text_input("Ridge Spacing")
    ridge_angle = st.text_input("Ridge Angle")
    flowrate = st.text_input("Fluid Flowrate")

# Dictionary to set fill and stroke colors for include and exlude polygon objects 
poly_type = {"Include":["rgba(10, 255, 0, 0.3)","rgba(10, 255, 0, 1)"],\
            "Exclude":["rgba(255, 10, 0, 0.3)","rgba(255, 10, 0, 1)"]}

output_colors = [(228,26,28),(55,126,184),(77,175,74),(152,78,163), \
    (255,127,0),(255,255,51),(166,86,40),(247,129,191),(153,153,153)]

# Draws Input and Output images if a background image has been loaded 
if spots_data and track_data and bg_image:

    # Read image dimensions and determine scaling factor for a width of 800 
    img_width,img_height = Image.open(bg_image).size
    scale_factor = img_width/800

    # Specify canvas parameters in application
    with st.sidebar.expander("Drawing Options"):
        poly_create = st.radio("Mode", ("Include","Exclude"))  
        group_name = st.text_input("Group Name", help="Add name for group")
        if st.button('Clear Groups'):
            initialize_session_state()

    # Initialize session state variables for iterative group creation
    # This prevents the groups from being deleted with each Streamlit run 
    if 'count' not in st.session_state:
        initialize_session_state()

    # Create drawing canvas using API
    st.write('### Input')
    canvas_result = st_canvas(
        fill_color=poly_type[poly_create][0],  
        stroke_width=1,
        stroke_color=poly_type[poly_create][1],
        background_color= "#eee",
        background_image=Image.open(bg_image),
        update_streamlit=True,
        height=img_height/scale_factor,
        width = 800,
        drawing_mode="polygon",
        point_display_radius=1,
        key="canvas"
    )

    # Creates new group when button is clicked (checks points, add labels to corresponding tracks, draws output)
    if st.button('Create Group') and len(canvas_result.json_data["objects"]) > 0:

        # Determine the number of polygons to check and get edge coefficients for each polygon 
        num_poly = len(canvas_result.json_data["objects"])
        edge_coefs = get_edge_coefs(canvas_result.json_data["objects"])

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
            if canvas_result.json_data["objects"][poly]["fill"] == poly_type["Include"][0]:
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

        # Increment the group ID
        st.session_state.count += 1

    with st.sidebar.expander("Display Settings"):
        show_stats = st.checkbox('Show Group Stats',False)
        items = list(st.session_state.groups.keys())
        output_options = ['All Groups','Ungrouped'] + items
        output_groups = st.selectbox("Select Group to Display",output_options)

    if st.sidebar.button('Export'):
        pass    

# Draw Output if background image is loaded 
if bg_image:
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

# NOTES:
# Add convexity checker
# Add export tracks button in dedicated expander ***
# Add metadata fields in dedicated expander or make it flexible ***
# Display Group metrics (number of tracks in group, aggregage stats, etc) 