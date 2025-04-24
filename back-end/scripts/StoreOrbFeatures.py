import cv2
import os
import json
from supabase import create_client, Client


# connect to supabase
url = "https://boiqldvzgaejmqxbsapi.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJvaXFsZHZ6Z2Flam1xeGJzYXBpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzAxNTk5OTIsImV4cCI6MjA0NTczNTk5Mn0.dNpXNFarQtMAFiAtUKZPJX5WopaHR3FEy8Ha4YU4GJQ"
supabase: Client = create_client(url, key)

# crawl our images, all of them
image_extensions = {'.jpg', '.jpeg', '.png', '.JPG'}

def insert_data_to_supabase(tablename, data):
    try:
        response = supabase.table(tablename).insert(data).execute()
        return response
    except Exception as e:
        print("error on image ", data['image_path'], ': ', str(e))
        return {"error":str(e)}

def check_extensions(directory):
    found_extensions = set()
    for root, _, files in os.walk(directory):
        print(f"Crawling in: {root}")
        for file in files:
            ext = os.path.splitext(file)[1]  # Get the file extension
            found_extensions.add(ext)
    print(found_extensions)

def convert_data_toJson(keypoints, des, path):
    keypoints_serialized = [
            {
                "pt": kp.pt,
                "size": kp.size,
                "angle": kp.angle,
                "response": kp.response,
                "octave": kp.octave,
                "class_id": kp.class_id,
            }
            for kp in keypoints
        ]
    
    des_serialized = des.tolist() if des is not None else None

    # Insert into Supabase
    data = {
        "image_path": path,
        "keypoints": json.dumps(keypoints_serialized),
        "descriptors": json.dumps(des_serialized),
    }

    return data

def compute_orb_features(path, orb):
    # read in the image
    im = cv2.imread(path, 0)
    # compute keypoint
    keypoint = orb.detect(im, None)
    # keypoints and des
    keypoints, des = orb.compute(im,keypoint)
    return keypoints, des

def crawl_files(directory, orb):
    for root, _, files in os.walk(directory):
        print(f"Crawling in: {root}")
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext in image_extensions:
                # print(f"Found image file: {file}")
                # get the file path
                path = os.path.join(root, file)
                # calculate ORB features and keypoints
                keypoints, des = compute_orb_features(path, orb)
                # store in supabase
                data = convert_data_toJson(keypoints, des, path)
                res = insert_data_to_supabase("OrbFeatures",data)
                # print(type(keypoints), type(des))
                print(res)
                


root = os.getcwd()
folderRoute = "nextjs-flask\\public\\duplicates"
basePath = os.path.join(root, folderRoute)
orb = cv2.ORB_create()
print(basePath)
crawl_files(basePath, orb)

# def test_insert():
#     data = {"Name":"this is my first string"}
#     try:
#         response = supabase.table("TestTable").insert(data).execute()
#         return response
#     except Exception as e:
#         return {"error":str(e)}

# def test_retrieve(columns: list = None, filters: dict = None):
#     supabase.auth.refresh_session()
#     try:
#         query = supabase.table("TestTable").select(",".join(columns) if columns else "*")
#         # Apply filters, if any
#         if filters:
#             for column, condition in filters.items():
#                 query = query.eq(column, condition)

#         # Execute the query
#         response = query.execute()
#         return response
#     except Exception as e:
#         return {"error": str(e)}
    
