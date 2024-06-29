#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#######################################################
# NLTK Inference
#######################################################
from nltk.sem import Expression
from nltk.sem import logic
from nltk.inference import ResolutionProver
read_expr = Expression.fromstring
#######################################################
import tensorflow as tf
from tensorflow import keras
#trained model
model = keras.models.load_model('yugioh_model.h5')
import tkinter as tk
from tkinter import filedialog

# TensorFlow and tf.keras
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
# ######
import numpy as np
import matplotlib.pyplot as pl
from PIL import Image
##asasdasdas###
import requests
from io import BytesIO
##
import matplotlib.pyplot as plt
#######################################################
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import csv
#######################################################
#AIML agent
#######################################################
import aiml
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="mybot-basic.xml")
#########################################################
#Knowledge Base
kb = []
with open('finalyugikb.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        kb.append(Expression.fromstring(row[0]))
#######################################################
# Welcome user
#######################################################
print("Welcome to the Yu-Gi-Oh! Chatbot. Please feel free to ask questions from me!")
#######################################################
# Main loop
#######################################################
while True:
    #get user input
    try:
        userInput = input("> ")
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break
    #pre-process user input and determine response agent (if needed)
    responseAgent = 'aiml'
    #activate selected response agent
    if responseAgent == 'aiml':
        answer = kern.respond(userInput)
        answer = answer.lower()
    #post-process the answer for commands
    
    if answer == "":
        
        yugioh_qa = pd.read_csv("yugiohQA.csv")
      
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(yugioh_qa['Question'])
   
        userInput_tfidf = vectorizer.transform([userInput])

        # cosine similarity between the userinput and all questions
        cosine_similarities = cosine_similarity(userInput_tfidf, X)

        max_sim_score = cosine_similarities.max()
        #print (max_sim_score) # delete this later
        threshold = 0.4  

        if max_sim_score < threshold:
            print("Sorry, I didn't quite understand.")
        else:
            # index of the most sim question
            most_sim_idx = cosine_similarities.argmax()

            most_sim_q = yugioh_qa.iloc[most_sim_idx]['Question']
            matching_answer = yugioh_qa.iloc[most_sim_idx]['Answer']

            print("Taken input as: ", most_sim_q)
            print("Answer: ", matching_answer) 
    
    elif answer[0] == '#':
        
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            
            print(params[1])
            break
        
        elif cmd == 10: # DEFINE * card_name
            cFrames = ["normal", "effect", "ritual", "fusion", "synchro", "xyz", "link",
                              "normal_pendulum", "effect_pendulum", "ritual_pendulum",
                              "fusion_pendulum", "synchro_pendulum", "xyz_pendulum"]
            succeeded = False
            api_url = "https://db.ygoprodeck.com/api/v7/cardinfo.php?name="
            full_url = api_url + params[1] # Combine URL and the card name
            #print("Full URL:", full_url)  #delete this later
            response = requests.get(full_url)
            if response.status_code == 200:
                response_json = response.json()
                if response_json["data"]:
                    card_info = response_json["data"][0]
            
                    # Extracting card info from json
                    card_name = card_info["name"]
                    card_type = card_info["type"]
                    card_description = card_info["desc"]
                    card_level = card_info.get("level", "Unknown")
                    card_attack = card_info.get("atk", "Unknown")
                    card_defense = card_info.get("def", "Unknown")
                    card_attribute = card_info.get("attribute", "Unknown")
                    card_frame = card_info.get("frameType")
  
                    
                    card_sets = [card_set["set_name"] for card_set in card_info["card_sets"]]
                        
                    # Card Info
                    print(card_name + " is a " + card_type)
                    print("Description: " + card_description)
                    
                    if card_frame in cFrames:
                        print("Level: " + str(card_level))
                        print("Attack: " + str(card_attack))
                        print("Defense: " + str(card_defense))
                        print("Attribute: " + card_attribute)
                    print("This card can be found in the following sets:")
                    for card_set in card_sets:
                        print("- " + card_set)
                        succeeded = True
            else:
                print("Failed to get card info:", response.status_code) #delete this later

            if not succeeded:
                print("Sorry, I could not find the card you requested")
        
        elif cmd == 11: #SHOW CARDS IN * ARCHETYPE
            succeeded = False
            archetype_name = params[1].replace("archetype", "").strip()
            api_url = f"https://db.ygoprodeck.com/api/v7/cardinfo.php?archetype={archetype_name}"
            response = requests.get(api_url)
    
            if response.status_code == 200:
                response_json = response.json()
                card_names = [card_info['name'] for card_info in response_json['data']]
        
                if card_names:
                    print(f"Cards in the {archetype_name} archetype:")
                    print("\n".join(card_names))
                    succeeded = True
                else:
                    print(f"No cards found in the {archetype_name} archetype.")
            else:
                print("Failed to get card info:", response.status_code)
    
            if not succeeded:
                print(f"Sorry, I could not find cards for the {archetype_name} archetype.")
            
                
        elif cmd == 12: #Show me the card
            succeeded = False
            card_name = params[1]
            api_url = f"https://db.ygoprodeck.com/api/v7/cardinfo.php?name={card_name}"
            response = requests.get(api_url)
            if response.status_code == 200:
                response_json = response.json()
                card_data = response_json['data'][0]  # Assuming only one card is returned
                if 'card_images' in card_data:
                    card_image_urls = [img['image_url'] for img in card_data['card_images']]
                    if card_image_urls:
                        print(f"Showing image for {card_name}:")
                        for url in card_image_urls:
                            # Get the image content from the URL
                            image_response = requests.get(url)
                            if image_response.status_code == 200:
                                # Open the image using PIL
                                img = Image.open(BytesIO(image_response.content))
                                # Show the image
                                img.show()
                                succeeded = True
                            else:
                                print(f"Failed to fetch image from URL: {url}")
                    else:
                         print("No image found for this card.")
                else:
                    print("No image information found for this card.")
            else:
                print("Failed to fetch card information:", response.status_code)

            if not succeeded:
                print("Sorry, I could not find the image for the card " + card_name)  

        elif cmd == 33: #What card type is this?
            print ("select image")
            root = tk.Tk()
            root.withdraw()  #
            file_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
            if file_path:
                print("Image file selected:", file_path)
                class_labels = ['effect','fusion', 'link', 'normal', 'spell','synchro', 'trap', 'xyz']
                
                #print("Class Labels:", class_labels)
            
                image = Image.open(file_path)

                # Resize the image to match the dimensions used for training (128x187)
                resized_image = image.resize((128, 187))

             
                resized_image_array = np.array(resized_image)

                # Display the resized image
                plt.imshow(resized_image_array)
                plt.axis('off')
                plt.show()
                
                preprocessed_image = resized_image_array / 255.0 
                # Reshape the image data to match the input shape 
                preprocessed_image = preprocessed_image.reshape(1, 187, 128, 3)  

                # Predict the card type
                predictions = model.predict(preprocessed_image)
                predicted_class_index = np.argmax(predictions)
                predicted_class_label = class_labels[predicted_class_index]

                print(" The Card type is :", predicted_class_label)
                
            else:
                print("No image file selected.")
            
        
            
            
        elif cmd == 31: # if input pattern is "I know that * is *"
            
           
            
        
            object,subject=params[1].split(' is ')
            expr=read_expr(subject + '(' + object + ')')
            
           
            # Initialize an empty list to store FOL expressions
                
            
            neg_expr = Expression.fromstring("-" + str(expr))
            contradiction = ResolutionProver().prove(neg_expr, kb)
            
            if contradiction:
                print("That statement contradicts with what i know.")
            else:
                kb.append(expr)
                print("OK, I will remember that")
             #   print(kb)
                
            
        elif cmd == 32:  
            object,subject=params[1].split(' is ')
            expr=read_expr(subject + '(' + object + ')')
            
           
            answer=ResolutionProver().prove(expr, kb)
            if answer is True:
                print('Correct.')
               # print(answer)
                
                
            elif answer is False:
                print('checking...')
              #  print (answer)
                neg_expr = Expression.fromstring("-" + str(expr))
                
                contradiction = ResolutionProver().prove(neg_expr, kb)
                
                if contradiction:
                    print("That is incorrect ")
                #    print(contradiction)
                else:
                    print("Sorry, i dont know!")
                 #   print (contradiction)
                
                
            elif answer is None:
                print("Sorry I don't know.")
                #print (answer)
    
            
                
        elif cmd == 34: # i know that attribute for x is y  
            
            object, subject = params[1].split(' for ')
            subject, attribute = subject.split(' is ')
          
    
            #print("Object:", object)
            #print("Subject:", subject)
            #print("Attribute:", attribute)

            expr = read_expr('attribute' + '(' + attribute + ', ' + subject + ')')
            #print (expr)
            
            neg_expr = Expression.fromstring("-" + str(expr))
            contradiction = ResolutionProver().prove(neg_expr, kb)
            
            if contradiction:
                print("That statement contradicts with what i know.")
            else:
                kb.append(expr)
                print("OK, I will remember that")
       
                
                
                
                
        elif cmd == 35:  # check that attribute for x is y  
            object, subject = params[1].split(' for ')
            subject, attribute = subject.split(' is ')
            
            expr = read_expr('attribute' + '(' + attribute + ', ' + subject + ')')
            
            
            answer=ResolutionProver().prove(expr, kb)
            if answer is True:
                print('Correct.')
                #print(answer)
                
                
            elif answer is False:
                print('checking...')
             #   print (answer)
                neg_expr = Expression.fromstring("-" + str(expr))
                
                contradiction = ResolutionProver().prove(neg_expr, kb)
                
                if contradiction:
                    print("That is incorrect ")
              #      print(contradiction)
                else:
                    print("Sorry, i dont know!")
                 #   print (contradiction)
                
                
            elif answer is None:
                print("Sorry I don't know.")
                #print (answer)        
            
        elif cmd == 99:
            print("I did not get that, please try again.")
            
    
    
    else:
        print(answer)