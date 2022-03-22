import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import time

# Sets the streamlit from the default narrow layout to the full width
st.set_page_config(layout="wide")

# ----------------------------------- FUNCTIONS -----------------------------------

# Functions to clean up TrackMate CSVs and convert to dataframes
@st.experimental_memo
def process_spots_data(spots_data):
    spots_df = pd.read_csv(spots_data)
    spots_df = spots_df.drop([0,1,2])
    spots_df = spots_df.reset_index()
    spots_array = np.array(spots_df[["POSITION_X", "POSITION_Y"]]).astype(float)
    return spots_df, spots_array

@st.experimental_memo
def process_track_data(track_data):
    track_df = pd.read_csv(track_data)
    track_df = track_df.drop([0,1,2])
    track_df = track_df.reset_index()
    return track_df

# Finds line coefficients for each polygon edge
def line_solver(x0, y0, x1, y1):
    a = float(y1 - y0)
    b = float(x0 - x1)
    c = - a*x0 - b*y0
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

# ----------------------------------- LAYOUT -----------------------------------

# Initialize session state variables for iterative group creation
# This prevents the groups from being deleted with each Streamlit run 
if 'count' not in st.session_state:
    st.session_state['count'] = 1
    st.session_state['groups'] = dict()
    st.session_state['selections'] = dict.fromkeys(['im','draw'])
    st.session_state.selections['im'] = Image.new('RGB', (1504, 352), (0,0,0))
    st.session_state.selections['draw'] = ImageDraw.Draw(st.session_state.selections['im'])

# Title and header
st.title("Tracking Data Labelling")
st.write("Annotate TrackMate tracking data with run information and track group labels")

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

with st.sidebar.expander('Video Metadata'):
    ridge_spacing = st.text_input("Ridge Spacing")
    ridge_angle = st.text_input("Ridge Angle")
    flowrate = st.text_input("Fluid Flowrate")

# Specify canvas parameters in application
with st.sidebar.expander("Group Parameters"):
    poly_create = st.radio("Drawing Mode", ("Include","Exclude"))  
    group_name = st.text_input("Group Name", help="Add name for group")

# Dictionary to set fill and stroke colors for include and exlude polygon objects 
poly_type = {"Include":["rgba(10, 255, 0, 0.3)","rgba(10, 255, 0, 1)"],\
            "Exclude":["rgba(255, 10, 0, 0.3)","rgba(255, 10, 0, 1)"]}

# Draws Input and Output images if a background image has been loaded 
if spots_data and track_data and bg_image:

    spots_df, spots_array = process_spots_data(spots_data)
    track_df = process_track_data(track_data)   

    # Create drawing canvas using API
    st.write('### Input')
    img_width,img_height = Image.open(bg_image).size
    canvas_result = st_canvas(
        fill_color=poly_type[poly_create][0],  
        stroke_width=1,
        stroke_color=poly_type[poly_create][1],
        background_color= "#eee",
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=True,
        height=img_height if bg_image else None,
        width=img_width if bg_image else None,
        drawing_mode="polygon",
        point_display_radius=1,
        key="canvas",
    )

    col1, col2 = st.columns([1,6])

    with col1:
        # Creates new group when button is clicked (checks points, add labels to corresponding tracks, draws output)
        if st.button('Create Group') and len(canvas_result.json_data["objects"]) > 0:

            # Determine the number of polygons to check and get edge coefficients for each polygon 
            num_poly = len(canvas_result.json_data["objects"])
            edge_coefs = get_edge_coefs(canvas_result.json_data["objects"])

            # Initialize positive and negative track sets 
            track_set = set()
            neg_set = set()

            # Iterate through all polygons and keep track of track IDs to add or subtract 
            # tic = time.perf_counter()
            for poly in range(num_poly):

                # returns index values of points included within polygon
                point_indices = test_all_points(spots_array,edge_coefs[poly])

                # retreive corresponding tracks for the bounded points
                poly_set = set(spots_df["TRACK_ID"][point_indices])

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
            draw_points = spots_df[spots_df['TRACK_ID'].isin(st.session_state.groups[group_id]['tracks'])]
            coords = tuple(zip(draw_points.POSITION_X.astype(float),draw_points.POSITION_Y.astype(float)))
            st.session_state.groups[group_id]['points'] = coords
            color = tuple(np.random.randint(150,255,3))
            st.session_state.selections['draw'].point(coords, fill=color)

            # Increment the group ID
            st.session_state.count += 1

            # toc = time.perf_counter()
            # st.write(toc - tic, "seconds")

    with col2:
        if st.button('Clear Groups'):
            st.session_state['count'] = 1
            st.session_state['groups'] = dict()
            st.session_state['selections'] = dict.fromkeys(['im','draw'])
            st.session_state.selections['im'] = Image.new('RGB', (1504, 352), (0,0,0))
            st.session_state.selections['draw'] = ImageDraw.Draw(st.session_state.selections['im'])

    with st.sidebar.expander("Output Options"):
        items = [str(key) for key in st.session_state.groups.keys()]
        output_options = ['All'] + items
        # if len(items) > 0:
        #     output_options.append(items)
        output_groups = st.selectbox("Select Group to Display",output_options)

# Draw Output if background image is loaded 
if bg_image:
    st.write('### Output')
    st.image(st.session_state.selections['im'])
    # st.write(st.session_state.groups)
    # st.write(list(output_options))

# NOTES:
# 1) Add convexity checker
# 2) Add better color picker
# 3) Add thicker spots
# 6) Add export tracks button in dedicated expander ***
# 7) Add metadata fields in dedicated expander ***
# 9) Add scaling function for input image (disable wide mode ) ***
# 10) Display Group metrics (number of tracks in group, aggregage stats, etc) ***
# 11) Flexible Video Metadata 
# 12) Custom video output display 