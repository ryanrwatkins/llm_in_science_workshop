import praw   # Reddit API
import json
import time
import os

from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser

import pw

# Change the working directory to the directory of the script so it saves there
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def reddit_comments():
	global posts_and_top_comments

	# Filename for storing the Reddit posts data
	filename = 'reddit_200_posts_1-20-25.json'

	# Check if the file exists and load data if it does
	if os.path.exists(filename):
		with open(filename, 'r') as f:
			posts_and_top_comments = json.load(f)
	else:

		# Reddit API credentials
		client_id = pw.client_id
		client_secret = pw.client_secret
		user_agent = 'PsychologyScriptApp'

		# Initialize the Reddit instance
		reddit = praw.Reddit(client_id=client_id,
				     client_secret=client_secret,
				     user_agent=user_agent,
				    check_for_async=False)

		# Select the subreddit
		subreddit = reddit.subreddit('AskPsychology')

		# Set the limit for the total number of questions and initialize counters
		total_questions = 200
		request_limit = 100
		posts_and_summaries = []
		request_count = 0


		# Fetch the top 10 submissions

		posts_and_top_comments = []
		for submission in subreddit.hot(limit=total_questions):  # Get the top 10 submissions
		    # Skip submissions that begin with "AskPsychology Weekly"
			if submission.title.startswith("AskPsychology Weekly"):
				continue
		    
			submission.comments.replace_more(limit=0)  # Remove MoreComments objects
		    
			if submission.comments:
				# Filter out AutoModerator comments
				filtered_comments = [comment for comment in submission.comments if comment.author != 'AutoModerator']
				# Sort comments by score and get the top 5
				top_comments = sorted(filtered_comments, key=lambda comment: comment.score, reverse=True)[:5]
				top_comments_texts = [comment.body for comment in top_comments]
			else:
				top_comments_texts = []

			posts_and_top_comments.append({
				"question": submission.title,
				"responses": top_comments_texts
		    	})
		    
			request_count += 1

		    	# Pause for 60 seconds after every 100 requests to respect rate limit
			if request_count % request_limit == 0:
				print(f"Processed {request_count} requests, pausing for 60 seconds...")
				time.sleep(60)

		# Optional -- Print the posts and the top comments in JSON format (this prints all, so just use for testing short list)
		json_output = json.dumps(posts_and_top_comments, indent=4)
		#print(json_output)

		# Save to a file
		with open('reddit_200_posts_1-20-25.json', 'w') as f:
		    json.dump(posts_and_top_comments, f, indent=4)





def llm_summaries():
	# Process each question and its responses
	output_data = []
	for item in posts_and_top_comments:
		question = item['question']
		responses = item['responses']
		prompt = PromptTemplate.from_template(
		"Objective: You are an experienced psychologist with years of training and your role is to articulate consencus pychological perspectives on various questions of college students. Your responses should reflect the complexity of these views highlighted in the responses that follow, with emphasizing principles of limited government, individual liberty, free market economics, traditional values, and national security without necessarily saying those are your principles. You are not necessarily religious, but you support peoples' right to freedom of religion and religious expression. Tone: Maintain a respectful and reasoned tone in all discussions. Your aim is to inform and persuade through logic, evidence, and principled arguments, rather than to provoke or antagonize. You are not combative, but you are trying to share evidence based psychological theories and ideas with students. Summarize in a conversational manner as a college age student, as if you were being asked the question in a classroom environment with high school seniors or college freshman.  Respond in the first person to make the conversation more authentic.  Limit responses to about 200 words. Your task is to write a summary of following responses,incorprating the comments into one final response. It does not have to be labeled. Here are the responses to summarize: \n\n# {input}"		)

		llm = Ollama(model="llama3:8b")      # tried with llama2:70b but it took ~15 seconds per summary and that was too long

		output_parser = StrOutputParser()
		chain = prompt | llm | output_parser
		result = chain.invoke(
		{"input": responses}
		)


		output_data.append({"from": "human", "value": question})
		output_data.append({"from": "gpt", "value": result})

	    	#print(f"Processed question: {question}")
	    	#print(f"Summary: {result}")


	#Optional Print the JSON output
	#json_output = json.dumps(output_data, indent=4)
	#print(json_output)

	# Save to a file
	with open('reddit_200_posts_summary_1-20-25.json', 'w') as f:
	    json.dump(output_data, f, indent=4)

def chatml():
	with open('reddit_200_posts_summary_1-20-25.json', 'r') as file:
		data = json.load(file)

	# Function to transform the dataset into pairs of human questions and GPT responses

	conversations = []
	for i in range(0, len(data), 2):
		if i + 1 < len(data):  # Ensure there is a pair
			conversation = {
				"conversations": [
				{"from": data[i]['from'], "value": data[i]['value']},
				{"from": data[i+1]['from'], "value": data[i+1]['value']}
				]
			}
			conversations.append(conversation)


	with open('reddit_200_posts_summary_1-20-21_chatml.json', 'w') as f:
		json.dump(conversations, f, indent=4)


if __name__ == "__main__":
	reddit_comments()
	llm_summaries()
	chatml()




# NEXT Unsloth tutorial: https://colab.research.google.com/drive/1XamvWYinY6FOSX9GLvnqSjjsNflxdhNc?usp=sharing#scrollTo=vITh0KVJ10qX