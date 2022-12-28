from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
import pickle
import face_recognition
import csv
from difflib import SequenceMatcher
import smtplib
# from tabulate import tabulate

# import pandas as pd

from pip._internal.utils.misc import tabulate

global capture,rec_frame, grey, switch, neg, face, rec, out
capture=0
grey=0
neg=0
face=0
switch=1
rec=0


#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

#Load pretrained face detection model
# net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt', './saved_model/res10_300x300_ssd_iter_140000.caffemodel')
# net = pickle.loads(open('Face_recognition\face_enc', "rb").read())

#instatiate flask app
app = Flask(__name__, template_folder='./templates')


camera = cv2.VideoCapture(0)

def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)

def Sorted(name):
    # csvData = pd.read_csv("IdentityResolution/csv_example_output.csv", delimiter=",")

    # csvData.sort_values(["Cluster ID"],
    #                     axis=0,
    #                     ascending=[True],
    #                     inplace=True)

    Csv = csv.reader(open("IdentityResolution/csv_example_output.csv", 'r', encoding='utf-8-sig'))
    Name_to_search = name
    List_of_similar_records = []

    for row in Csv:
        if len(row) == 0:
            continue
        else:
            s = SequenceMatcher(None, Name_to_search, row[3])
            if s.ratio() >= .50:
                List_of_similar_records.append(row)

    List_of_similar_records = sorted(List_of_similar_records)
    return List_of_similar_records


def detect_face(frame):

    cascPathface = os.path.dirname(
        cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    # load the harcaascade in the cascade classifier

    faceCascade = cv2.CascadeClassifier(cascPathface)

    # load the known faces and embeddings saved in last file

    data = pickle.loads(open('Face_recognition/face_enc', "rb").read())
    # data2 = pickle.loads(open('Face_recognition/FinalData.csv', "wb").read())

    print("Streaming started")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=4,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)



    # convert the input frame from BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # the facial embeddings for face in input
    encodings = face_recognition.face_encodings(rgb)
    names = []
    count=0
    with open("Face_recognition/FinalData.csv", 'w', encoding='utf-8-sig') as f_output:
            writer = csv.DictWriter(f_output,fieldnames=["name", "count"])
            writer.writeheader()

    for encoding in encodings:
        # Compare encodings with encodings in data["encodings"]
        # Matches contain array with boolean values and True for the embeddings it matches
        # and False for rest
        matches = face_recognition.compare_faces(data["encodings"],
                                                 encoding)
        # set name =unknown if no encoding matches

        name = "Unknown"
        # check to see if we have found a match
        if True in matches:
            # Find positions at which we get True and store them
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:

                name = data["names"][i]

                counts[name] = counts.get(name, 0) + 1
            # set name which has highest count
            # print(name)
            # name = max(counts, key=counts.get)

        # update the list of names
        names.append(name)
        # if name=='Unknown' and count<=0:
        #     count+=1
        #     val = name + count
        #     writer.writerow(val)
        #     continue
        # loop over the recognized faces

        for ((x, y, w, h), name) in zip(faces, names):

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            frame = cv2.flip(frame, 1)

    cv2.imshow("Frame", frame)



    return frame


def gen_frames():
    global out, capture, rec_frame
    while True:
        success, frame = camera.read()
        if success:
            if (face):
                frame = detect_face(frame)
            if (rec):
                rec_frame = frame
                frame = cv2.putText(cv2.flip(frame, 1), "Recording...", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 4)
                frame = cv2.flip(frame, 1)

            if (capture):
                capture = 0
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":", ''))])
                cv2.imwrite(p, frame)

            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
                # ret, buffer = cv2.imencode('.jpg')
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
        else:
            pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera
    if request.method == 'POST':
        if request.form.get('click') == 'Take Photo':
            global capture
            capture = 1

        elif request.form.get('face') == 'Detect Face':
            global face
            face = not face
            if (face):
                time.sleep(4)
        elif request.form.get('getname') == "Get Suspects":
            # print("hello")
            Name = request.form.get('name')
            # print(Name)
            server = smtplib.SMTP("smtp.gmail.com",587)
            server.starttls()
            server.login("anup.21bce7985@vitapstudent.ac.in","mqcopphjfzaitcll")
            Data = Sorted(Name)
            Data_str = " "
            for a in Data:
                for i in a:
                    Data_str += i +  " "
                Data_str +=  "\n"
            # print(type(Data_str))
            server.sendmail("anup7970hm@gmail.com","anup7970hm@gmail.com",Data_str.encode('utf-8-sig'))

            server.quit()

            return render_template("Data.html",content = tabulate(Data))

        elif request.form.get('stop') == 'Stop and Start':

            if (switch == 1):
                switch = 0
                camera.release()
                cv2.destroyAllWindows()
            else:
                camera = cv2.VideoCapture(0)
                switch = 1
        elif request.form.get('rec') == 'Recording':
            global rec, out
            rec = not rec
            if (rec):
                now = datetime.datetime.now()
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('vid_{}.mp4'.format(str(now).replace(":", '')), fourcc, 20.0, (640, 480))
                thread = Thread(target=record, args=[out, ])
                thread.start()
            elif (rec == False):
                out.release()

    elif request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)