import os
from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional
import google.generativeai as genai
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="AI Travel Planner API",
    description="A FastAPI application for AI-powered travel planning using Google Gemini",
    version="1.0.0"
)

# Configure your Gemini API key directly in the code
GEMINI_API_KEY = "AIzaSyCpugWq859UTT5vaOe01EuONzFweYT2uUY"

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
def get_gemini_model():
    return genai.GenerativeModel('gemini-1.5-pro')

# Routes
@app.get("/generate_plan")
async def generate_plan(
    destination: str,
    duration: int = Query(5, ge=1, le=30),
    budget: str = Query("Moderate", enum=["Budget", "Moderate", "Luxury"]),
    travel_styles: str = Query("Culture,Nature")
):
    try:
        # Parse travel styles
        styles_list = travel_styles.split(',')
        
        # Get Gemini model
        model = get_gemini_model()
        
        # Generate ultra-concise travel plan
        prompt = f"""Create an EXTREMELY BRIEF travel plan for {destination} for {duration} days.

Travel Preferences:
- Budget: {budget}
- Styles: {', '.join(styles_list)}

FORMAT YOUR RESPONSE AS FOLLOWS - USE ONLY ONE LINE PER SECTION:

WHEN: [only list best months, max 5 words]
STAY: [name 1-2 top {budget} accommodations only]
DO: [list only 1 must-see activity per day, numbered by day]
EAT: [name 1-2 dishes and 1 restaurant only]
TIPS: [only 2 critical tips - transport and budget]

TOTAL LENGTH MUST BE UNDER 200 WORDS. Use extreme brevity - single words or short phrases only.
NO explanations, NO descriptions, NO links, NO introductions or conclusions.
"""
        
        # Set parameters to enforce extreme brevity
        generation_config = {
            "temperature": 0.1,  # Very low temperature for predictable responses
            "max_output_tokens": 512,  # Very limited token count
            "top_p": 0.95,
        }
        
        # Generate the response with the specific configuration
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        return {"status": "success", "travel_plan": response.text}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating travel plan: {str(e)}")

@app.get("/ask_question")
async def ask_question(
    destination: str,
    question: str,
    travel_plan: str
):
    try:
        # Get Gemini model
        model = get_gemini_model()
        
        # Process question with extreme brevity instructions
        context_question = f"""
        Travel plan for {destination}:
        {travel_plan}
        
        Question: {question}
        
        Answer in ONE SENTENCE ONLY. Maximum 15 words. Just facts, no explanations.
        """
        
        # Set parameters to enforce extreme brevity
        generation_config = {
            "temperature": 0.1,
            "max_output_tokens": 128,
        }
        
        response = model.generate_content(
            context_question,
            generation_config=generation_config
        )
        
        return {"status": "success", "answer": response.text}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting answer: {str(e)}")

# Run the application
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
