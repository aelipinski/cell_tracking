import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import time

st.set_page_config(layout="wide")

def create_dot(x,y):
    dot = {
    "type": "circle",
    "version": "4.4.0",
    "originX": "left",
    "originY": "center",
    "left": x,
    "top": y,
    "width": 2,
    "height": 2,
    "fill": "rgba(10, 0, 255, 0.3)",
    "stroke": "rgba(10, 0, 255, 1)",
    "strokeWidth": 1,
    "strokeDashArray": None,
    "strokeLineCap": "butt",
    "strokeDashOffset": 0,
    "strokeLineJoin": "miter",
    "strokeUniform": False,
    "strokeMiterLimit": 4,
    "scaleX": 1,
    "scaleY": 1,
    "angle": 0,
    "flipX": False,
    "flipY": False,
    "opacity": 1,
    "shadow": None,
    "visible": True,
    "backgroundColor": "",
    "fillRule": "nonzero",
    "paintFirst": "fill",
    "globalCompositeOperation": "source-over",
    "skewX": 0,
    "skewY": 0,
    "radius": 1,
    "startAngle": 0,
    "endAngle": 6.283185307179586
    }
    return dot


spots = pd.read_csv("spots.csv")
spots = spots.drop([0,1,2])
# spots_filt = spots[spots["TRACK_ID"] == "2"]
spots_filt = spots
spots_filt = spots_filt.reset_index()
# print("Display")
# print(len(spots_filt))

spots_array = np.array(spots_filt[["POSITION_X", "POSITION_Y"]]).astype(float)

spots_init = {"version": "4.4.0","objects":[]}

for k in range(len(spots_array)):
    x = spots_array[k,0]
    y = spots_array[k,1]
    spots_init["objects"].append(create_dot(x,y))
    

# Specify canvas parameters in application
poly_create = st.sidebar.radio(
    "Drawing tool:", ("Include","Exclude")
)

bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

poly_type = {"Include":["rgba(10, 255, 0, 0.3)","rgba(10, 255, 0, 1)"],\
            "Exclude":["rgba(255, 10, 0, 0.3)","rgba(255, 10, 0, 1)"]}

# canvas_result = st_canvas(
#     fill_color=poly_type[poly_create][0],  # Fixed fill color with some opacity
#     stroke_width=1,
#     stroke_color=poly_type[poly_create][1],
#     background_color= "#eee",
#     background_image=Image.open(bg_image) if bg_image else None,
#     update_streamlit=True,
#     height=400,
#     width=2000,
#     drawing_mode="polygon",
#     point_display_radius=0,
#     key="canvas",
# )


if bg_image:
    st.write('### Input')
    img_width,img_height = Image.open(bg_image).size
    canvas_result = st_canvas(
        fill_color=poly_type[poly_create][0],  # Fixed fill color with some opacity
        stroke_width=1,
        stroke_color=poly_type[poly_create][1],
        background_color= "#eee",
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=True,
        height=352 if bg_image else None,
        width=1504 if bg_image else None,
        drawing_mode="polygon",
        # initial_drawing=spots_init2 if draw else None,
        # drawing_mode="point",
        # point_display_radius=0,
        point_display_radius=1,
        key="canvas",
    )
    # st.write(canvas_result.json_data)
    # st.write(type(canvas_result.json_data))
    # st.write(len(spots_init2["objects"]))


# realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Create a canvas component
# canvas_result2 = st_canvas(
#     fill_color=poly_type[poly_create][0],  # Fixed fill color with some opacity
#     stroke_width=1,
#     stroke_color=poly_type[poly_create][1],
#     background_color= "#eee",
#     background_image= None,
#     update_streamlit=True,
#     height=352,
#     width=1504,
#     drawing_mode="point",
#     initial_drawing=spots_init,
#     point_display_radius=3,
#     key="canvas2",
# )

# def test_point(x, y, vertices):
#     num_vert = len(vertices)
#     is_right = [is_on_right_side(x, y, vertices[i], vertices[(i + 1) % num_vert]) for i in range(num_vert)]
#     all_left = not any(is_right)
#     all_right = all(is_right)
#     return all_left or all_right

# Do something interesting with the image data and paths
# if canvas_result.image_data is not None:
#     st.image(canvas_result.image_data)
# if canvas_result.json_data is not None:
#     objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
#     for col in objects.select_dtypes(include=['object']).columns:
#         objects[col] = objects[col].astype("str")
#     st.dataframe(objects)
#     st.write(canvas_result.json_data["objects"])

# @st.cache(suppress_st_warning=True,show_spinner=True)

# def line_solver(x0, y0, x1, y1):
#     a = float(y1 - y0)
#     b = float(x0 - x1)
#     c = - a*x0 - b*y0
#     return (a,b,c)

def line_solver2(x0, y0, x1, y1):
    a = float(y1 - y0)
    b = float(x0 - x1)
    c = - a*x0 - b*y0
    return np.array([a,b,c])

# def get_line_coefs(canvas_objects):
#     all_coefs = {}
#     for i in range(len(canvas_objects)):
#         poly = canvas_result.json_data["objects"][i]["path"]
#         num_vert = len(poly) - 1 
#         line_coefs = {edge:line_solver(poly[edge][1], poly[edge][2], poly[(edge + 1) % num_vert][1], poly[(edge + 1) % num_vert][2]) for edge in range(num_vert)} 
#         #is_on_right_side2(x, y, poly[i][1], poly[i][2], poly[(i + 1) % num_vert][1], poly[(i + 1) % num_vert][2]) for i in range(num_vert)}
#         all_coefs[i] = line_coefs
#     return all_coefs

def get_edge_coefs(canvas_objects):
    num_poly = len(canvas_objects)
    poly_array = {}
    for i in range(num_poly):
        poly = canvas_result.json_data["objects"][i]["path"]
        num_vert = len(poly) - 1 
        coef_mat = np.zeros((3,num_vert))
        for edge in range(num_vert):
            coef_mat[:,edge] = line_solver2(poly[edge][1], poly[edge][2], poly[(edge + 1) % num_vert][1], poly[(edge + 1) % num_vert][2])
        poly_array[i] = coef_mat
    return poly_array


# def is_right_side(x, y, coefs):
#     a,b,c = coefs 
#     return a*x + b*y + c >= 0

# def test_point(x, y, all_coefs):
#     is_right = [is_right_side(x, y, all_coefs[poly]) for poly in all_coefs]
#     all_left = not any(is_right)
#     all_right = all(is_right)
#     return all_left or all_right

def test_all_points(spots_array, all_coefs):
    prod_mat = spots_array @ all_coefs[:2,:] + all_coefs[2,:] 
    sideA = prod_mat<=0
    sideB = ~sideA
    point_indices = np.all(sideA, axis=1) + np.all(sideB, axis=1)
    # point_indices = np.where(np.all(sideA, axis=1) + np.all(sideB, axis=1))
    # st.write(point_indices)
    return point_indices

if 'count' not in st.session_state:
    st.session_state['count'] = 1
    st.session_state['groups'] = dict()
    st.session_state['selections'] = dict.fromkeys(['im','draw'])
    st.session_state.selections['im'] = Image.new('RGB', (1504, 352), (0,0,0))
    st.session_state.selections['draw'] = ImageDraw.Draw(st.session_state.selections['im'])

# st.write(st.session_state)
if st.sidebar.button('Create Group'):

    num_poly = len(canvas_result.json_data["objects"])
    edge_coefs = get_edge_coefs(canvas_result.json_data["objects"])

    track_set = set()
    neg_set = set()
    tic = time.perf_counter()
    for poly in range(num_poly):
        # returns index values of points included within polygon
        point_indices = test_all_points(spots_array,edge_coefs[poly])

        # retreive corresponding tracks for the bounded points
        poly_set = set(spots_filt["TRACK_ID"][point_indices])

        # Check if polygon is an include or exclude type by looking at it's color (red or green)
        # Then add or remove track IDs depending on type
        if canvas_result.json_data["objects"][poly]["fill"] == poly_type["Include"][0]:
            track_set = track_set | poly_set
        else:
            neg_set = neg_set | poly_set
    track_set = track_set - neg_set

    group_id = st.session_state.count
    st.session_state.groups[group_id] = track_set
    st.session_state.count += 1

    
    # for group in st.session_state.groups:
    #     draw_points = spots_filt[spots_filt['TRACK_ID'].isin(st.session_state.groups[group])]
    #     coords = tuple(zip(draw_points.POSITION_X.astype(float),draw_points.POSITION_Y.astype(float)))
    #     color = tuple(np.random.randint(150,255,3))
    #     draw.point(coords, fill=color)


    draw_points = spots_filt[spots_filt['TRACK_ID'].isin(st.session_state.groups[group_id])]
    coords = tuple(zip(draw_points.POSITION_X.astype(float),draw_points.POSITION_Y.astype(float)))
    color = tuple(np.random.randint(150,255,3))
    st.session_state.selections['draw'].point(coords, fill=color)

    # st.image(im)

    # st.write(st.session_state.groups)

    toc = time.perf_counter()
    # st.write(toc - tic, "seconds")


if bg_image:
    st.write('### Output')
    st.image(st.session_state.selections['im'])
