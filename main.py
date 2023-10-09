from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

class SalaryInput(BaseModel):
    YearsExperience: float

@app.get("/")
def home():
    return {"message": "ML model for Salary prediction"}

@app.post("/predict/")
async def predict_salary(data: SalaryInput):
    years_experience = data.YearsExperience
    salary_prediction = model.predict([[years_experience]])[0]
    return {"YearsExperience": years_experience, "PredictedSalary": salary_prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
