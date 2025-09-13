from transformers import pipeline

# Load the QA model
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Load your transcript
with open("transcript.txt", "r", encoding="utf-8") as f:
    transcript = f.read()

while True:
    question = input("Ask a question (or type 'exit' to quit): ")
    if question.lower() == "exit":
        break
    
    result = qa_pipeline(question=question, context=transcript)
    print(f"\nAnswer: {result['answer']}\n")
