from telethon import TelegramClient, sync

#from tqdm import  tqdm
import time

from datetime import datetime, timezone
import pytesseract
from pytesseract import Output
import cv2
import numpy as np
import sys
import os
import requests

from os import listdir
from os.path import isfile, join


def count_files(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    return len(onlyfiles)

sys.stdout = open(os.devnull, "w")

def read_text(image):
    d = pytesseract.image_to_string(image, output_type=Output.DICT)
    return d

myobj = {'key': 'saadi_bot123456788',
         'sport':'AM FOOTBALL',
         'event':'CAROLINA PANTHERS - DALLAS COWBOYS',
         'bet':'CAROLINA PANTHERS -+AH (LIVE)    or    DALLAS COWBOYS -+AH (LIVE)',
         'odds':1.50,
         'stake':2,
         'book':'PINNACLE',
         'showid':1
         }



sys.stdout = sys.__stdout__
url = 'https://postserver.smartbet.io/postpick'
api_id = 1681517
api_hash = "2531ada92c27f808c3fb4fb59f0332bd"

prev_sessions_path = "temp_data\\previous_sessions.csv"
bets_data_path = "temp_data\\bets_logs.csv"

class bet():

    def __init__(self,type,teams,bets,odds,bet_team,direction):

        self.type = type
        self.teams = teams
        self.bets = bets
        self.odds = odds
        self.bet_team = bet_team
        self.direction = direction


def send_bet_post(current_session,current_bet,image_name):

    if (current_bet.type =='Asian Handicap'):

        team1 = current_bet.teams[0].upper()
        team2 = current_bet.teams[1].upper()
        myobj = {'key': current_session.bot_id,
                 'sport': current_session.sport,
                 'event': team1+' - '+team2,
                 'bet': current_bet.bet_team.upper()+" "+current_bet.bets[0]+','+current_bet.bets[1]+" "+"(LIVE)",
                 'odds': float(current_bet.odds),
                 'stake': current_session.stakes,
                 'book': 'BET365',
                 'showid': 1
                 }

        print(myobj)
        print("\nSending post request....")
        x = requests.post(url, data=myobj)

        # print the response text (the content of the requested file):


        print("Status: ",x.text)
    if (current_bet.type == 'Total 2-Way'):
        team1 = current_bet.teams[0].upper()
        team2 = current_bet.teams[1].upper()
        myobj = {'key': current_session.bot_id,
                 'sport': current_session.sport,
                 'event': team1 + ' - ' + team2,
                 'bet': current_bet.direction.upper()+" "+current_bet.bets[0]+" (LIVE)",
                 'odds': float(current_bet.odds),
                 'stake': current_session.stakes,
                 'book': 'BET365',
                 'showid': 1
                 }
        print(myobj)
        print("\nSending post request....")
        x = requests.post(url, data=myobj)

        # print the response text (the content of the requested file):

        print("Status: ", x.text)
    if (current_bet.type == 'Full Time Result'):
            team1 = current_bet.teams[0].upper()
            team2 = current_bet.teams[1].upper()
            myobj = {'key': current_session.bot_id,
                     'sport': current_session.sport,
                     'event': team1 + ' - ' + team2,
                     'bet': current_bet.bet_team.upper(),
                     'odds': float(current_bet.odds),
                     'stake': float(current_session.stakes),
                     'book': 'BET365',
                     'showid': 1
                     }
            print(myobj)
            print("\nSending post request....")
            x = requests.post(url, data=myobj)

            # print the response text (the content of the requested file):

            print("Status: ", x.text)
            save_bet_info(current_session,current_bet,x.text.split(":")[1].strip(),image_name)


def save_bet_info(session,bet,bet_id,bet_image):

    file = open(bets_data_path,'a+')
    time_stamp = datetime.now(timezone.utc)
    file.write(session.name+","+session.sport+","+session.stake+","+session.bot_id+","+bet.type+","+ str(bet_image) +","+str(time_stamp)+","+bet_id)
    print("Bet details noted...")
    file.close()

def refresh_sessions_details(sessions):
    file = open(prev_sessions_path, 'w+')
    for i in range(len(sessions)):
        session = sessions[i]
        file.write(session.name + "," +session.channel + "," + session.sport + "," + session.stakes  + "," + session.bot_id + "\n")
    file.close()

    print("Sessions details updated.")

def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Grab the image size and initialize dimensions
    dim = None
    (h, w) = image.shape[:2]

    # Return original image if no need to resize
    if width is None and height is None:
        return image

    # We are resizing height if width is none
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # We are resizing width if height is none
    else:
        # Calculate the ratio of the 0idth and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Return the resized image
    return cv2.resize(image, dim, interpolation=inter)

def read_text(image):
    d = pytesseract.image_to_string(image, output_type=Output.DICT)
    return d


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def read_team_info(cropped):
    (cropped_h,cropped_w) = cropped.shape
    info = {}

    d = pytesseract.image_to_data(cropped, output_type=Output.DICT)
    n_boxes = len(d['level'])

    info['team'] = ''
    cut_percentage = 0.738
    for i in range(n_boxes):

            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])

            if(x> cut_percentage*cropped_w ):
                info['odds'] = float(d['text'][i])

            else:
                if (is_number(d['text'][i])):
                    info['bet'] = float(d['text'][i])

                else:
                    info['team'] += " "+d['text'][i]
    info['team'] = info['team'].strip()
    info['team'] =' '.join(info['team'].split())
    return info

def find_clr(img):
    data = np.reshape(img, (-1, 3))
    # print(data.shape)
    data = np.float32(data)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(data, 1, None, criteria, 10, flags)

    # print(sum(centers[0])/(3))
    return (sum(centers[0]) / (3))


def process_big_image(orig_image):
    img = orig_image.copy()
    (image_height, image_width, image_channels) = (orig_image.shape)

    green = (0, 255, 0)
    red = (0, 0, 255)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (T, img) = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    ( cnts,__) = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)

    my_box = None
    for i in range(len(cnts)):
        (x,y,w,h) = cv2.boundingRect(cnts[i])

        if (w > 0.75*image_width and y > 0.1*image_height and (h>0.15*w and h<0.25*w) ):
            #print("box found")
            my_box = orig_image[y:y+h,x:x+w]
            cv2.rectangle(orig_image, (x, y), (x + w, y + h), red, 2)
    data_info = process_main_box(my_box)
    return [data_info]

    #print("Bet details:\n",data_info.type,data_info.teams,data_info.odds,data_info.bets,data_info.bet_team)
def process_small_image(orig_image):
    img = orig_image.copy()
    (image_height, image_width, image_channels) = (orig_image.shape)

    green = (0, 255, 0)
    red = (0, 0, 255)

    my_box = img
    #cv2.imshow("Threshold image:",img)

    data_info = process_main_box(my_box)
    #print("Bet details:\n", data_info.type, data_info.teams, data_info.odds, data_info.bets, data_info.bet_team)
    #cv2.waitKey(0)
    return [data_info]

def process_main_box(my_box):
  #cv2.imshow("My box",my_box)
  #print(my_box.shape)
  cv2.imwrite("wrong_working_image.jpg", my_box)

  my_box = maintain_aspect_ratio_resize(my_box, height=int(1.5 * my_box.shape[0]))
  my_box = cv2.cvtColor(my_box, cv2.COLOR_BGR2GRAY)
  (T, my_box) = cv2.threshold(my_box, 190, 255, cv2.THRESH_BINARY)

  text = read_text(my_box)['text']
  lines = text.split("\n")
  #print("\nResults:\n")
  line1 = lines[0].split(" ")

  direction = None
  if ('Over' in line1 or 'Under' in line1):
        print(line1)
        print("Bsdk OVER/UNDER kahaan mill gya?")
        direction = line1[0]
        bet_value = line1[1]
        odds = line1[2]
        type = 'Total 2-Way'
        last_line = lines[len(lines) - 1]
        if (" v " in last_line):
            teams = last_line.split(" v ")
        elif (" @ " in last_line):
            teams = last_line.split(" @ ")

        teams[0] = teams[0].strip()
        teams[1] = teams[1].strip()
        #type,teams,bets,odds,bet_team):
        this_bet = bet(type,teams,[bet_value],odds,None,direction)
        return this_bet
  else:



    if('(' in line1[0] and ')' in line1[0]):
        bet_team = line1[1:len(line1) - 2]
        bet_team = (" ".join(bet_team)).strip()
    else:
        bet_team = line1[0:len(line1) - 2]
        bet_team = (" ".join(bet_team)).strip()

    second_last_scores = line1[len(line1) - 2].split(",")

    scores = []
    if (len(second_last_scores) == 2):
        scores = [line1[len(line1) - 2].split(",")[0], line1[len(line1) - 2].split(",")[1], line1[len(line1) - 1]]
    elif (len(second_last_scores) == 1):
        scores = [line1[len(line1) - 2].split(",")[0], line1[len(line1) - 1]]
    #print("Parameters: ", scores)

    last_line = lines[len(lines) - 1]
    if (" v " in last_line):
        teams = last_line.split(" v ")
    elif (" @ " in last_line):
        teams = last_line.split(" @ ")

    teams[0] = teams[0].strip()
    teams[1] = teams[1].strip()
    #print("Bet team: ",bet_team)


    #print("Teams: ", teams)
    #cv2.imwrite("original_image.jpg", orig_image)
    #cv2.imshow("Cropped part", my_box)
    #cv2.imshow("Original", orig_image)
    data_info = {'parameters':scores , 'teams': teams}
    type = None
    bets = []
    odds = data_info['parameters'][len(data_info['parameters'])-1]
    if (len(data_info['parameters']) < 3):
        type = 'Full Time Result'

    else:
        type = 'Asian Handicap'
        bets = [data_info['parameters'][0],data_info['parameters'][1]]

    #type,teams,bets,odds,bet_team):
    this_bet = bet(type,teams,bets,odds,bet_team,None)

    return this_bet

def find_text(image,text):
    locations = []
    d = pytesseract.image_to_data(image, output_type=Output.DICT)
    n_boxes = len(d['level'])


    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        #process_small_image(d[text][i])
        if(d['text'][i]== text):
            locations.append([x,y,w,h])

    return locations

def show_bet(bet):
    print("----------------------\nType: ",bet.type,"\nTeams: ",bet.teams,"\nBet-team: ",bet.bet_team,"\nBets: ",bet.bets,"\nOdds: ",bet.odds,"\nDirection: ",bet.direction,"\n----------------------")

def process_last_test_image(orig_image):
    img = orig_image.copy()
    (image_height, image_width, image_channels) = (orig_image.shape)

    green = (0, 255, 0)
    red = (0, 0, 255)

    my_box = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
    (T, my_box) = cv2.threshold(my_box, 215, 255, cv2.THRESH_BINARY)
    text_locations = (find_text(my_box,'Stake'))
    print(text_locations)
    bets = process_last_testbox(orig_image,text_locations)
    #for i in range(len(bets)):
        #print(i+1, bets[i].type , bets[i].teams, bets[i].odds, bets[i].bet_team,bets[i].bets)
    return bets
    #cv2.imshow("Threshold image:",my_box)
    #cv2.waitKey(0)

def process_last_testbox(image,text_locations):

    up_offset = 10
    lower_offset = 120

    bets = []
    for i in range(len(text_locations)):
        (x,y,w,h) = text_locations[i]

        cropped = image[ y - up_offset : y+h+lower_offset , 0:x]
        cropped_rgb = cropped.copy()
        cropped = maintain_aspect_ratio_resize(cropped,width=int(1.5*cropped.shape[1]))
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        (T, cropped) = cv2.threshold(cropped, 200, 255, cv2.THRESH_BINARY)
        d = pytesseract.image_to_data(cropped, output_type=Output.DICT)
        n_boxes = len(d['level'])
        this_bet = None
        text = ""
        for j in range(n_boxes):
            (x, y, w, h) = (d['left'][j], d['top'][j], d['width'][j], d['height'][j])

            # process_small_image(d[text][i])
            if ( x > 0.68 * image.shape[1]):
                #print("Right side")
                odds = float(d['text'][j])
                #print("Odds: ",odds)
            else:
               text+=" "+d['text'][j]
            #cv2.rectangle(cropped,(x,y),(x+w,y+h),(0,0,255),1)


        lines = text.split("  ")
        while ("" in lines):
            lines.remove("")
        teams = lines[len(lines)-2]+lines[len(lines)-1]
        if (" v " in teams):
            teams = teams.split(" v ")
        if (" @ " in teams):
            teams = teams.split(" @ ")

        if( len( lines[1].split(",")) == 2):
            this_bet = bet('Asian Handicap',teams,lines[1].split(","),odds,lines[0].strip(),None)
            bets.append(this_bet)

        else:
            this_bet = bet('Full Time Result', teams, None, odds, lines[0].strip(),None)
            bets.append(this_bet)


        #print("Lines: ", lines)
        #cv2.imshow("Cropped", cropped)
        #cv2.waitKey(0)

    return(bets)

def ask_user(options):
    done = False
    selection = None
    while (not done):
        for i in range(len(options)):
            print(i+1," - ",options[i])

        print(0," - ","Exit\n---------------------")
        selection = input("Selection: ")

        if (is_number(selection) and int(selection)>-1 and int(selection)<=len(options)):
            done = 1
            return int(selection)
        else:
            done = 0
            print("Invalid choice, please try again.")


class session():

    def __init__(self,name,channel,sport,stakes,bot_id):
        self.name = name

        self.channel = channel
        self.stakes = stakes
        self.bot_id = bot_id
        self.sport = sport

    def set_sport(self,sport):
        self.sport = sport

    def set_bot_id(self,bot_id):
        self.bot_id = bot_id
    def set_channel(self,channel):
        self.channel = channel
    def set_stakes(self,stakes):
        self.stakes = stakes


def load_previous_sessions(path):
    temp_file = open(path,'r')
    lines = temp_file.readlines()
    prev_sessions = []
    for i in range(len(lines)):
        if (len(lines[i])>5):
            line = lines[i].split(",")
        #    name, channel ,sport, stakes,  bot-ID
            sess = session(line[0].strip(),line[1].strip(),line[2].strip(),line[3].strip(),line[4].strip())
            prev_sessions.append(sess)
    temp_file.close()
    return prev_sessions

def read_msgs(client,channel_name,n):
    return client.get_messages(channel_name, n)

def show_prev_sessions(prev_sessions):
    # Loading prev_sessions
    print(
        "Previous sessions:\nS. No., Name,   Channel  ,       Sport      , Stakes , Bot-ID\n------------------")
    for i in range(len(prev_sessions)):
        sess = prev_sessions[i]
        print(i+1,sess.name+","+ sess.channel+","+sess.sport+","+ sess.stakes+","+  sess.bot_id)

    print("-----------------------------------------------------------------")

def start_session(session):
    client = TelegramClient(session.name, api_id, api_hash).start()


    last_timestamp = datetime.now(timezone.utc)
    print("Checking.....")
    count = count_files("media\\")
    while (1):

        chats = read_msgs(client,session.channel,5)
        # Get message id, message, sender id, reply to message id, and timestamp
        chats = reversed(chats)

        for msg in chats:


                if (msg.date > last_timestamp):
                    print("\nReceived new msg:")

                    time_stamp = msg.date
                    last_timestamp = time_stamp
                    print("Time:", time_stamp)
                    name = "bet_image" + str(count) + ".jpg"
                    try:
                        count += 1

                        print("Media location: ", msg.download_media(file=os.path.join('media', name)))

                        img = cv2.imread('media\\'+name)

                        img = maintain_aspect_ratio_resize(img, width=591)
                        (h,w,c) = img.shape
                        my_box = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        (T, my_box) = cv2.threshold(my_box, 215, 255, cv2.THRESH_BINARY)
                        locs = find_text(my_box,'Stake')
                        if (len(locs)>0):
                            results = process_last_test_image(img)
                        else:
                            if (h>0.4*w):
                                #print("Here")
                                results = process_big_image(img)
                            else:
                                results = process_small_image(img)
                        for i in range(len(results)):
                            show_bet(results[i])
                            send_bet_post(session,results[i],name)



                    except Exception as e:
                        print("Ooopss... Something is wrong. Could not process this image. Error has been logged. Please contact your developer.")
                        logger = open("Error_logs\\error_logs.txt",'a+')
                        logger.write("\n--------------\n")
                        logger.write(str(e)+'\n')
                        logger.write("Image: "+name)
                        logger.write(str(datetime.now(timezone.utc)))
                        logger.close()

                        cv2.imwrite("Error_logs\\images\\"+name,img)

                    #input("Press Enter to continue....")
                    #cv2.imshow("image:",img)
                    #cv2.waitKey(0)

        time.sleep(1)

print("Welcome...")
while(1):
    selection = ask_user(["Create a new Session.","Load a previous session."])

    if (selection == 1):
        # creating a new session...
        print("Create a new session: \n")
        session_name = input("Session name: ").strip()
        channel_id = input("Link to the channel: ").strip()
        stakes = input("Stakes: ").strip()
        bot_id = input("Bot-ID: ").strip()
        sport = input("Sport: ").strip()
        print("Saving the info to database...")
        file = open(prev_sessions_path,'a+')
        file.write(session_name+","+channel_id+","+sport+","+stakes+","+bot_id+"\n")
        file.close()

        print("Data saved successfully.")
        session = session(session_name,channel_id,sport,stakes,bot_id)
        input("Press Enter to begin this session.")
        start_session(session)

    if (selection == 2):


        prev_sessions = load_previous_sessions(prev_sessions_path)
        show_prev_sessions(prev_sessions)

        serial_number = int(input("Serial Number: "))
        session = prev_sessions[serial_number-1]

        print("Selected session: ",session.name)

        channel = input("Channel link: (Press ENTER if you want to use the default: ("+session.channel+") : ")
        if (channel != ""):
            session.set_channel(channel.strip())

        sport = input("Sport: (Press ENTER if you want to use the default: (" + session.sport + ") : ")
        if (sport != ""):
            session.set_sport(sport.strip())

        stakes = input("Stake: (Press ENTER if you want to use the default: (" + session.stakes + ") : ")
        if (stakes != ""):
            session.set_stakes(stakes.strip())

        bot_id = input("Bot ID (KEY): (Press ENTER if you want to use the default: (" + session.bot_id + ") : ")
        if (bot_id != ""):
            session.set_bot_id(bot_id.strip())


        prev_sessions[serial_number-1] = session

        refresh_sessions_details(prev_sessions)



        input("Press ENTER to start the session.....")
        start_session(session)

    if selection == 0:
        print("Program ended...")
        break


