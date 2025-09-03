import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize the FastAPI App
app = FastAPI(
    title="Toxicity Multi-Label Classifier API",
    description="Classify a comment across multiple toxicity categories using a pre-trained model.",
    version="1.0",
)

# Load the Pretrained Model and Tokenizer
try:
    MODEL_NAME = "unitary/unbiased-toxic-roberta"  # Replace with a suitable multi-label classification model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    toxicity_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
    print("Model loaded successfully!")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Define Input Data Model for API Request
class CommentRequest(BaseModel):
    comment: str  # Input comment to be classified


# Health Check Endpoint
@app.get("/")
def health_check():
    """Health check endpoint to ensure the service is running."""
    return {"message": "Toxicity Multi-Label Classifier API is running!"}


# API Endpoint for Classifying Comments
@app.post("/classify/")
def classify_comment(request: CommentRequest):
    try:
        if not request.comment.strip():
            raise HTTPException(status_code=400, detail="Comment cannot be empty.")

        # Get raw classification results
        results = toxicity_classifier(request.comment)

        # Log raw results
        print("Raw Model Results:", results)

        # Define thresholds and initialize response
        thresholds = {
            'toxic': 0.5,
            'severe_toxic': 0.5,
            'obscene': 0.5,
            'threat': 0.5,
            'insult': 0.3,
            'identity_hate': 0.5,
            'identity_attack':0.3,
            'sexual_explicit':0.3,
            'psychiatric_or_mental_illness':0.5,
        }
        response = {key: False for key in thresholds}

        # Process model results
        for category_scores in results:
            for score_info in category_scores:
                label = score_info["label"].lower()
                score = score_info["score"]
                if label in thresholds:
                    print(f"Label: {label}, Score: {score}")  # Debug log
                if label in thresholds and score >= thresholds[label]:
                    response[label] = True

        return {
            "comment": request.comment,
            "categories": response,
        }
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


# Run the FastAPI App with Auto-Reload for Development
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002, reload=True)



    steps to run the code

    step 1 create vertual environment in terminal
           type --->   python -m venv venv

    step 2 activate the vertual environment
          type --->  .\venv\Scripts\activate 
    
    step 3 install dependiences
          type --->  pip install fastapi uvicorn transformers torch  

    step 4 open host in terminal
         type ---> uvicorn app:app --reload --host 0.0.0.0 --port 8000  

    step 5 run localhost in chrome or any search engine
         type--> http://127.0.0.1:8000/docs


    step 6 check the app is running aur not in health_check part using try it out  
         if app is running then give the the comment in post part clicking on try it out
    check the toxicity in Response body
    download the result 

    step 7  to close the app 
          type--> ctrl+c

   step 8 to close the vertual environment
          type --> deactivate

        