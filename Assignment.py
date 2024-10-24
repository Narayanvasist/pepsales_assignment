import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
import spacy
import numpy as np
from textblob import TextBlob
from pymongo import MongoClient

# Load the spaCy English model
nlp = spacy.load('en_core_web_sm')

# MongoDB connection setup
client = MongoClient('mongodb://localhost:27017/')
db = client['questions_db']  # Database
collection = db['questions']  # Collection

# 1. Extract text from PDFs
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# 2. Find questions using regex and tokenize
def find_questions(text):
    questions = re.findall(r'(\w+.*\?)', text)
    return questions

# 3. Identify similar questions using TF-IDF and cosine similarity
def find_similar_questions(questions):
    vectorizer = TfidfVectorizer().fit_transform(questions)
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    similar_questions = []
    
    for idx, row in enumerate(cosine_sim):
        similar = np.where(row > 0.8)[0]  # Threshold for similarity (can be tuned)
        if len(similar) > 1:
            similar_questions.append((questions[idx], len(similar)))
    
    # Sort by number of similar questions
    similar_questions.sort(key=lambda x: x[1], reverse=True)
    return similar_questions

# 4. Extract answers corresponding to top questions (next sentence after a question)
# 4. Extract answers corresponding to top questions (next sentence after a question)
def extract_answers(text, questions):
    # Process the entire text
    doc = nlp(text)
    # Split text into sentences
    sentences = [sent.text for sent in doc.sents]
    qa_pairs = []
    
    # Regex pattern to detect time stamps (e.g., '45:10 |')
    timestamp_pattern = re.compile(r'\d{1,2}:\d{2}(?=\s?\|)')

    # Find answers corresponding to questions (assuming answers follow questions)
    for question in questions:
        for i, sentence in enumerate(sentences):
            if question in sentence:
                # Assuming the answer is the next sentence
                if i + 1 < len(sentences):
                    answer = sentences[i + 1]
                    # Skip the answer if it matches a time stamp pattern
                    if timestamp_pattern.search(answer):
                        if i + 2 < len(sentences):
                            answer = sentences[i + 2]  # Get the next valid sentence
                    qa_pairs.append((question, answer))
                break
    return qa_pairs


# 5. Rate the answers based on length and sentiment (can be customized)
def rate_answers(answers):
    rated_answers = []
    for question, answer in answers:
        sentiment = TextBlob(answer).sentiment.polarity
        if sentiment > 0.5:
            rating = "Best"
        elif 0 < sentiment <= 0.5:
            rating = "Good"
        else:
            rating = "Average"
        rated_answers.append((question, answer, rating))
    return rated_answers

# 6. Store in MongoDB and Print
def store_in_mongo(rated_answers):
    print("\nTop 5 Questions, Answers, and Ratings:\n")
    for question, answer, rating in rated_answers[:5]:  # Store top 5
        data = {
            "question": question,
            "answer": answer,
            "rating": rating
        }
        collection.insert_one(data)
        
        # Print the top 5 questions, answers, and ratings
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print(f"Rating: {rating}")
        print("\n" + "-"*50 + "\n")
    
    print("Top 5 questions and answers stored in MongoDB")

# 7. Retrieve and display questions and answers from MongoDB
def display_from_mongo():
    results = collection.find()
    for result in results:
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Rating: {result['rating']}")
        print("\n" + "-"*50 + "\n")

# 8. Write all questions and stored ones to text files with reasons
def write_questions_to_file(questions, rated_answers):
    # Write all extracted questions
    with open('all_questions.txt', 'w', encoding='utf-8') as file:
        file.write(f"Total Questions Extracted: {len(questions)}\n\n")
        file.write("All Extracted Questions:\n\n")
        for question in questions:
            file.write(f"{question}\n")
    
    # Write stored questions and reasons for storing them
    with open('stored_questions.txt', 'w', encoding='utf-8') as file:
        file.write("Stored Questions and Reasons:\n\n")
        for question, answer, rating in rated_answers:
            reason = f"Stored because it was rated as '{rating}' based on sentiment analysis."
            file.write(f"Question: {question}\nAnswer: {answer}\nRating: {rating}\nReason: {reason}\n\n")

# Additional Outputs:
def generate_additional_outputs(questions, similar_questions, rated_answers):
    # 1. Print total number of extracted questions
    print(f"Total number of extracted questions: {len(questions)}")

    # 2. Print total number of similar questions found
    print(f"Total number of similar questions found: {len(similar_questions)}")

    # 3. Print top 5 questions based on the highest ratings
    print("\nTop 5 Questions Based on Ratings:\n")
    rated_answers.sort(key=lambda x: x[2], reverse=True)  # Sort by rating
    for question, answer, rating in rated_answers[:5]:
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print(f"Rating: {rating}")
        print("\n" + "-"*50 + "\n")

# Main Execution
pdf_paths = ["C:\\Repo\\documents\\abc_call_1.pdf", "C:\\Repo\\documents\\abc_call_2.pdf", "C:\\Repo\\documents\\abc_call_3.pdf","C:\\Repo\\documents\\abc_call_4.pdf", "C:\\Repo\\documents\\abc_call_6.pdf", "C:\\Repo\\documents\\abc_call_7.pdf"]
all_text = ""

for pdf_path in pdf_paths:
    all_text += extract_text_from_pdf(pdf_path)

# Extract all questions from the text
questions = find_questions(all_text)

# Ensure unique questions for top 5
questions = list(set(questions))

# Find the top similar questions
similar_questions = find_similar_questions(questions)

# Get top 5 questions based on similarity count
top_5_questions = [q[0] for q in similar_questions[:5]]

# Extract corresponding answers
answers = extract_answers(all_text, top_5_questions)

# Rate the answers based on sentiment
rated_answers = rate_answers(answers)

# Store top 5 questions and answers in MongoDB and print the output
store_in_mongo(rated_answers)

# Retrieve and display the stored questions from MongoDB
print("\nStored Questions and Answers from MongoDB:\n")
display_from_mongo()

# Write all extracted questions and stored ones with reasons to text files
write_questions_to_file(questions, rated_answers)

# Generate additional outputs (Total questions, Total similar, Top rated)
generate_additional_outputs(questions, similar_questions, rated_answers)
