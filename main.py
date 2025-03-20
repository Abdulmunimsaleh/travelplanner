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
        
        # Generate travel plan
        prompt = f"""Create a comprehensive travel plan for {destination} for {duration} days.

Travel Preferences:
- Budget Level: {budget}
- Travel Styles: {', '.join(styles_list)}

Please provide a detailed itinerary that includes:

1. 🌞 Best Time to Visit
- Seasonal highlights
- Weather considerations

2. 🏨 Accommodation Recommendations
- {budget} range hotels/stays
- Locations and proximity to attractions

3. 🗺️ Day-by-Day Itinerary
- Detailed daily activities
- Must-visit attractions
- Local experiences aligned with travel styles

4. 🍽️ Culinary Experiences
- Local cuisine highlights
- Recommended restaurants
- Food experiences matching travel style

5. 💡 Practical Travel Tips
- Local transportation options
- Cultural etiquette
- Safety recommendations
- Estimated daily budget breakdown

6. 💰 Estimated Total Trip Cost
- Breakdown of expenses
- Money-saving tips

Please provide source and relevant links for accommodations, attractions, and restaurants.

Format the response in a clear, easy-to-read markdown format with headings and bullet points.
        """
        
        response = model.generate_content(prompt)
        
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
        
        # Process question
        context_question = f"""
        I have a travel plan for {destination}. Here's the existing plan:
        {travel_plan}

        Now, please answer this specific question: {question}
        
        Provide a focused, concise answer that relates to the existing travel plan if possible.
        """
        
        response = model.generate_content(context_question)
        
        return {"status": "success", "answer": response.text}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting answer: {str(e)}")

# Run the application
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
