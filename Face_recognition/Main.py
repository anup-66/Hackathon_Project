# import face_recognition
# import imutils
# import pickle
# import time
# import cv2
# import os
#
# # find path of xml file containing haarcascade file
# cascPathface = os.path.dirname(
#     cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
# # load the harcaascade in the cascade classifier
# faceCascade = cv2.CascadeClassifier(cascPathface)
# # load the known faces and embeddings saved in last file
# data = pickle.loads(open('face_enc', "rb").read())
#
# print("Streaming started")
# video_capture = cv2.VideoCapture(0)
# warning = 0
# # loop over frames from the video file stream
# while True:
#     # grab the frame from the threaded video stream
#     ret, frame = video_capture.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = faceCascade.detectMultiScale(gray,
#                                          scaleFactor=1.1,
#                                          minNeighbors=6,
#                                          minSize=(60, 60),
#                                          flags=cv2.CASCADE_SCALE_IMAGE)
#     print(faces ,"            " , faces)
#     # res = all(isinstance(ele, list) for ele in faces)
#     # print(res)
#     count = 0
#
#     # res = all(isinstance(ele, list) for ele in faces[0])q
#     # for i in faces:
#     #     count += 1
#     # print(count)
#     # if(warning>1):
#     #     exit("Sry more than members detected ur test is temporary banned")
#     # if(count>1):
#     #     print("hey we have detected more than two members in ur screen this is the last warning>>>>>>>")
#     #     warning+=1
#
#     # convert the input frame from BGR to RGB
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     # the facial embeddings for face in input
#     encodings = face_recognition.face_encodings(rgb)
#     names = []
#     # loop over the facial embeddings incase
#     # we have multiple embeddings for multiple faces
#     counttts = 0
#     for encoding in encodings:
#         # Compare encodings with encodings in data["encodings"]
#         # Matches contain array with boolean values and True for the embeddings it matches closely
#         # and False for rest
#         matches = face_recognition.compare_faces(data["encodings"],
#                                                  encoding)
#         # set name =unknown if no encoding matches
#
#         name = "Unknown"
#         # check to see if we have found a match
#         if True in matches:
#             # Find positions at which we get True and store them
#             matchedIdxs = [i for (i, b) in enumerate(matches) if b]
#             counts = {}
#             # loop over the matched indexes and maintain a count for
#             # each recognized face face
#             for i in matchedIdxs:
#                 # Check the names at respective indexes we stored in matchedIndex
#                 name = data["names"][i]
#                 # increase count for the name we got
#                 counts[name] = counts.get(name, 0) + 1
#             # set name which has highest count
#             name = max(counts, key=counts.get)
#
#
#         # update the list of names
#         names.append(name)
#         # loop over the recognized faces
#
#         for ((x, y, w, h), name) in zip(faces, names):
#             # rescale the face coordinates
#             # draw the predicted face name on the image
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
#                         0.75, (0, 255, 0), 2)
#
#     cv2.imshow("Frame", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# video_capture.release()
# cv2.destroyAllWindows()

# a = "punA"
# b = a[::-1]
# print(b)