from __future__ import print_function
from credentials import *
from time import sleep
from textblob import TextBlob
from PIL import Image, ImageStat
from numpy.random import choice
from numpy import interp
from datetime import datetime
import os
import sys
import time
import random
import decimal
import tweepy
import wget
import time
import json
import re
import threading
import nltk
import markovify
import neuralart


#Handles Twitter credentials and declares Twitter operations
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

#creates advanced learning model for Markovify
class POSifiedText(markovify.Text):
    def word_split(self, sentence):
        words = re.split(self.word_split_pattern, sentence)
        words = [ "::".join(tag) for tag in nltk.pos_tag(words) ]
        return words

    def word_join(self, words):
        sentence = " ".join(word.split("::")[0] for word in words)
        return sentence

def write_JSON(models):
#write models to JSON files

    json_models = []
    for index, model in enumerate(models):
        json_models.append(model.to_json())
        with open("./" + str(index) + ".txt", "w") as outfile:
            json.dump(json_models[index], outfile)

    print("Successfully written to JSON files")

def read_JSON(num_texts):
#read and create models from existing JSON files

    index = 0
    models = []
    while index < num_texts:
        exists = os.path.isfile(str(index) + ".txt")
        if exists:
            with open(str(index) + ".txt") as json_file:
                model_json = json.load(json_file)
                models.append(POSifiedText.from_json(model_json))
        index+=1

    return models
    

#Begins neural network and sentiment analysis
def generate_text(tweet):
#generates text using Markov chains

    num_texts = 0    
    with open("./the_picture_of_dorian_gray.txt", encoding="utf8") as f:
        the_picture_of_dorian_gray = f.read()
        num_texts+=1
    with open("./declaration_of_independence.txt", encoding="utf8") as f2:
        declaration_of_independence = f2.read()
        num_texts+=1
    with open("./jane_eyre.txt",encoding = "utf8") as f3:
        declaration_of_independence = f3.read()
        num_texts+=1
    with open("./war_and_peace.txt", encoding="utf8") as f4:
        war_and_peace = f4.read()
        num_texts+=1
    with open("./jane_eyre.txt", encoding="utf8") as f5:
        jane_eyre = f5.read()
        num_texts+=1
    with open("./jokes.txt", encoding="utf8") as f6:
        jokes = f6.read()
        num_texts+=1
    with open("./moby_dick.txt", encoding="utf8") as f7:
        moby_dick = f7.read()
        num_texts+=1
    with open("./pride_and_prejudice.txt", encoding="utf8") as f8:
        pride_and_prejudice = f8.read()
        num_texts+=1
    with open("./art_of_love.txt", encoding="utf8") as f9:
        art_of_love = f9.read()
        num_texts+=1

    state_size = 4
    max_overlap_ratio = 50
    num_tries = 100

    models = read_JSON(num_texts)
    model_tweet = POSifiedText(tweet.full_text, state_size = state_size)
    if len(models) == 0:
        print("Creating language model")
        #creates custom models instead of using naive ones
        model_a = POSifiedText(the_picture_of_dorian_gray, state_size = state_size)
        model_b = POSifiedText(declaration_of_independence, state_size = state_size)
        model_c = POSifiedText(jane_eyre, state_size = state_size)
        model_d = POSifiedText(war_and_peace, state_size = state_size)
        model_e = POSifiedText(jokes, state_size = state_size)
        model_f = POSifiedText(pride_and_prejudice, state_size = state_size)
        model_g = POSifiedText(moby_dick, state_size = state_size)
        model_h = POSifiedText(art_of_love, state_size = state_size)
        

        models = []
        models.append(model_a)
        models.append(model_b)
        models.append(model_c)
        models.append(model_d)
        models.append(model_e)
        models.append(model_f)
        models.append(model_g)
        models.append(model_h)

        write_JSON(models)

    else:
        print("Sucessfully created models from existing JSON files")
    
    model_combo = markovify.combine([models[0], models[1], models[2], models[3], models[4], models[5], models[6], models[7], model_tweet], [1.4, 1.25, 1.25, 1.4, 1.25, 1.0, 1.25, 2.0, 1.0])
    return model_combo.make_short_sentence(280, max_overlap_ratio = max_overlap_ratio, tries = num_tries)
    
def generate_image(tweet, contains_photos, polarity, subjectivity):
#generates image by feeding sentiment analysis results into a neural network
    
    #modes = {"1": 1, "L": 8, "P": 8, "RGB": 24, "RGBA": 32, "CMYK": 32, "YCbCr": 24, "LAB": 24, "HSV": 24, "I": 32, "F": 32}

    if contains_photos is True: #when a user's tweet contains photos
        image_files = []
        if 'media' in tweet.entities:
            for image in tweet.entities['media']:
                media_url = str(image['media_url'])
                image_files.append(media_url)
                
        for image in image_files:
            wget.download(image, "download_image.png")
                                   
        seed_path = "download_image.png"
        seed_image = Image.open(seed_path)
        im_obj = ImageStat.Stat(seed_image)
        
        UNITS = int(subjectivity*2+10)
        DEPTH = int(interp(polarity, [-1,1],[3,13]))
        
        if seed_image.mode == "RGB":
            CHANNELS = 3
            RENDER_SEED = int(interp(subjectivity, [0,1],[3,40]))
            Z_DIMS = int((im_obj.stddev[0] + im_obj.stddev[1] + im_obj.stddev[2])/3)
        else:
            CHANNELS = 1
            RENDER_SEED = int(interp(subjectivity, [0,1],[3,200]))
            min_max = seed_image.getextrema()
            #Z_DIMS = random.randint(1, min_max[1] - min_max[0])
            Z_DIMS = int(interp(len(tweet.full_text), [1,280],[1,100]))
        os.remove(seed_path) #remove the downloaded photo

    else: #when a user's tweet contains text only
        UNITS = int(subjectivity*2+10)
        DEPTH = int(interp(polarity, [-1,1],[3,13]))
        CHANNELS = 1
        RENDER_SEED = int(interp(subjectivity, [0,1],[3,40]))
        #Z_DIMS = int(interp(subjectivity, [0,1],[1,40]))
        Z_DIMS = int(interp(len(tweet.full_text), [1,280],[1,100]))

    ITERATIONS = 10
    #full-hd resolution
    WIDTH = 1920
    HEIGHT = 1080
    HIDDEN_STD = float(random.randrange(100, 500))/100
    OUTPUT_STD = 1.0
    RADIUS = True
    BIAS = True
    zfill = len(str(ITERATIONS - 1))
    z = [-1.0] * Z_DIMS
    step_size =  1/ITERATIONS #the higher the step size, the quicker the pattern transformation 

    print("")
    print("----------------RENDERING PARAMETERS-------------------")
    print("SEED: {}".format(RENDER_SEED))
    print("DEPTH: {}".format(DEPTH))
    print("WIDTH: {}".format(WIDTH))
    print("HEIGHT: {}".format(HEIGHT))
    print("CHANNELS: {}".format(CHANNELS))
    print("Z-DIMENSIONS: {}".format(Z_DIMS))
    print("STEP SIZE: {}".format(step_size))
    print("UNITS: {}".format(UNITS))
    print("HIDDEN STD: {}".format(HIDDEN_STD))
    print("OUTPUT STD: {}".format(OUTPUT_STD))
    print("RADIUS: {}".format(RADIUS))
    print("BIAS: {}".format(BIAS))
    print("ITERATIONS: {}".format(ITERATIONS))
    print("")

    
    generated_images = []
    #feeds values into the neural network
    for x in range(ITERATIONS):
        result = neuralart.render(
            xres = WIDTH,
            yres = HEIGHT,
            seed = RENDER_SEED,
            channels = CHANNELS,
            hidden_std = HIDDEN_STD,
            output_std = OUTPUT_STD,
            units = UNITS,
            depth = DEPTH,
            device = 'cpu',
            radius = RADIUS,
            bias = BIAS,
            z=z   
        )
        im = Image.fromarray(result.squeeze())
        generated_images.append(im)
        print("Completed iteration {}".format(x+1))

    buckets = []
    probability_distribution = []
    for i in range(0,len(generated_images)):
        buckets.append(i)

    for i, value in enumerate(generated_images):
        if i > len(generated_images)/2:
            probability_distribution.append(i/len(generated_images))
        else:
            probability_distribution.append(0)

    chosen_image_id = int(choice(buckets, 1, probability_distribution))
    
    generated_images[len(generated_images)-1].save("generated.png", 'png')    

def analyze_tweet(tweet):
#analyzes a tweet to see if it contains photos or not
    
    contains_photos = False
    try:
        print(True in [medium['type'] == 'photo' for medium in tweet.entities['media']])
        contains_photos = True
    except:
        print("No picture in this tweet")

    #calculates polarity and subjectivity
    testimonial = TextBlob(tweet.full_text)
    polarity = testimonial.sentiment.polarity
    subjectivity = testimonial.sentiment.subjectivity
    print("Polarity value: {}. Possible range: [-1,1]".format(polarity))
    print("Subjectivity value: {}. Possible range: [0,1]".format(subjectivity))

    return contains_photos, polarity, subjectivity

def respond(tweet):
#Tweets with both a photo and text
    
    contains_photos, polarity, subjectivity = analyze_tweet(tweet)
    text = generate_text(tweet)
    print("Generated text: {}".format(text))
    generate_image(tweet, contains_photos, polarity, subjectivity)
    tweet_image("generated.png", text, tweet)
    print("Response tweeted")


def tweet_image(filename, message, tweet):
#tweets the image along with message
    reply_status = "@{} {}".format(tweet.user.screen_name, message)
    api.update_with_media(filename, status=reply_status, in_reply_to_status = tweet.id)
    os.remove(filename) #removes the generated image    

def utc_to_local(utc_datetime):
#converts utc time to local time
    
    now = time.time()
    offset = datetime.fromtimestamp(now) - datetime.utcfromtimestamp(now)
    return utc_datetime + offset

def follow_back():
#follows back everyone who follows the bot
    
    followers = tweepy.Cursor(api.followers).items()

    for follower in followers:
        follower.follow()

def main():
#uses multi-threading to listen on the entire Twitter user space for mentions
    #nltk.download('averaged_perceptron_tagger')
    time_now = datetime.now()
    tweets = []
    known_users = []
    cold_start = True
    wait_time = 60
    
    #passive mode
    while True:
        if cold_start == True: #cold start so load every tweet into list
            for tweet in tweepy.Cursor(api.search, q = "@Soul_Snapshot", tweet_mode = 'extended').items(10):
                tweets.append(tweet)
            
        else: #only load non-available tweets
            for tweet in tweepy.Cursor(api.search, q = "@Soul_Snapshot", tweet_mode = 'extended').items(10):
                if tweet not in tweets:
                    tweets.append(tweet)

        cold_start = False
        count = 0
        thread_list = []
        for tweet in tweets:
            post = tweet
            post_time = utc_to_local(post.created_at) #tweet.created at returns UTC timezone
            if post_time > time_now: #only responds to new tweets
                print("Now responding to user {}, who tweeted: {}, at {}".format(tweet.user.screen_name, tweet.full_text, post_time))
                t = threading.Thread(target = respond, args = (post,))
                thread_list.append(t)
                t.start()
                count+=1
                follow_back() #follows back everyone who followed upon start up
            
        for thread in thread_list:
            thread.join()
            
        if count == 0:
            print("No new interaction detected. Sleep for {} seconds and check again".format(wait_time))
        else:
            print("Responded to all new tweets. Sleep for {} seconds and check again".format(wait_time))
            time_now = datetime.now() #reset the time to current
                               
        sleep(wait_time) #wait to comply with Twitter's rate limit

if __name__ == "__main__":
    main()


