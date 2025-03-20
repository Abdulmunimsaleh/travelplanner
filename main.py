import os
from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.serpapi_tools import SerpApiTools
import uvicorn
from dotenv import load_dotenv

# Initialize FastAPI app
app = FastAPI(
    title="AI Travel Planner API",
    description="A FastAPI application for AI-powered travel planning",
    version="1.0.0"
)

# Configure your API keys directly in the code
GROQ_API_KEY = "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"
SERP_API_KEY = "ffd21e065fb0e3817b366c3f0fad1026f132969bef6eba499b2774aab2e03920"

# Initialize the travel agent
def create_travel_agent():
    # Set API keys
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    os.environ["SERP_API_KEY"] = SERP_API_KEY
    
    # Initialize travel agent
    return Agent(
        name="Travel Planner",
        model=Groq(id="llama-3.3-70b-versatile"),
        tools=[SerpApiTools()],
        instructions=[
            "You are a travel planning assistant using Groq Llama.",
            "Help users plan their trips by researching destinations, finding attractions, suggesting accommodations, and providing transportation options.",
            "Give me relevant live Links of each places and hotels you provide by searching on internet (It's important)",
            "Always verify information is current before making recommendations."
        ],
        show_tool_calls=True,
        markdown=True
    )

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
        
        # Create travel agent
        travel_agent = create_travel_agent()
        
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

Please provide source and relevant links without fail.

Format the response in a clear, easy-to-read markdown format with headings and bullet points.
        """
        
        response = travel_agent.run(prompt)
        
        if hasattr(response, 'content'):
            clean_response = response.content.replace('∣', '|').replace('\n\n\n', '\n\n')
            return {"status": "success", "travel_plan": clean_response}
        else:
            return {"status": "success", "travel_plan": str(response)}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating travel plan: {str(e)}")

@app.get("/ask_question")
async def ask_question(
    destination: str,
    question: str,
    travel_plan: str
):
    try:
        # Create travel agent
        travel_agent = create_travel_agent()
        
        # Process question
        context_question = f"""
        I have a travel plan for {destination}. Here's the existing plan:
        {travel_plan}

        Now, please answer this specific question: {question}
        
        Provide a focused, concise answer that relates to the existing travel plan if possible.
        """
        
        response = travel_agent.run(context_question)
        
        if hasattr(response, 'content'):
            return {"status": "success", "answer": response.content}
        else:
            return {"status": "success", "answer": str(response)}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting answer: {str(e)}")

# Run the application
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
