from fastapi import FastAPI, HTTPException
from gold_analysis import run_analysis_and_predict, GoldPriceAnalyzer

app = FastAPI(
    title="Gold Price Analysis API",
    description="An API to get gold price analysis and predictions, connected to AkwaMining.",
    version="1.0.0"
)

@app.get("/api/v1/predict/latest", tags=["Predictions"])
async def get_latest_prediction():
    """
    Runs the analysis model on the latest data and returns the prediction.
    """
    try:
        result = run_analysis_and_predict()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during analysis: {str(e)}")

@app.get("/api/v1/models/comparison", tags=["Analysis"])
async def get_model_comparison():
    """
    Compare different prediction models performance.
    """
    try:
        analyzer = GoldPriceAnalyzer()
        analyzer.load_data()
        comparison = analyzer.get_model_comparison()
        return comparison
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during model comparison: {str(e)}")

@app.get("/", tags=["Status"])
def root():
    """Root endpoint to check if the API is online."""
    return {"status": "ok", "message": "Welcome to the Gold Price Analysis API. See /docs for documentation."}