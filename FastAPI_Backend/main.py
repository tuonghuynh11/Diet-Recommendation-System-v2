from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
from model import recommend, output_recommended_recipes

# dataset = pd.read_csv('../Data/dataset.csv', compression='gzip')
dataset = pd.read_csv('../Data/dataset.csv', compression='gzip')

app = FastAPI()

class NutritionInput(BaseModel):
    Calories: float = Field(2000, gt=0, le=2000, description="Từ 0 đến 2000 kcal")
    FatContent: float = Field(100, gt=0, le=100, description="Từ 0 đến 100 g")
    SaturatedFatContent: float = Field(13, gt=0, le=13, description="Từ 0 đến 13 g")
    CholesterolContent: float = Field(300, gt=0, le=300, description="Từ 0 đến 300 mg")
    SodiumContent: float = Field(2300, gt=0, le=2300, description="Từ 0 đến 2300 mg")
    CarbohydrateContent: float = Field(325, gt=0, le=325, description="Từ 0 đến 325 g")
    FiberContent: float = Field(50, gt=0, le=50, description="Từ 0 đến 50 g")
    SugarContent: float = Field(40, gt=0, le=40, description="Từ 0 đến 40 g")
    ProteinContent: float = Field(40, gt=0, le=40, description="Từ 0 đến 40 g")

class Params(BaseModel):
    n_neighbors: int = 5
    return_distance: bool = False

class PredictionIn(BaseModel):
    nutrition_input: NutritionInput
    ingredients: List[str] = []
    params: Optional[Params] = None

class Recipe(BaseModel):
    id: str = Field(..., alias='_id')
    Name: str
    Image: str
    CookTime: str
    PrepTime: str
    TotalTime: str
    RecipeIngredientQuantities: List[str]
    RecipeIngredientParts: List[str]
    Calories: float
    FatContent: float
    SaturatedFatContent: float
    CholesterolContent: float
    SodiumContent: float
    CarbohydrateContent: float
    FiberContent: float
    SugarContent: float
    ProteinContent: float
    RecipeInstructions: List[str]

class PredictionOut(BaseModel):
    output: Optional[List[Recipe]] = None

@app.get("/")
def home():
    return {"health_check": "OK"}

@app.post("/diet/recommend", response_model=PredictionOut)
def update_item(prediction_input: PredictionIn):
    # Chuyển NutritionInput thành list[float] để đưa vào hàm recommend
    nutrition_list = list(prediction_input.nutrition_input.dict().values())
    recommendation_dataframe = recommend(
        dataset,
        nutrition_list,
        prediction_input.ingredients,
        prediction_input.params.dict() if prediction_input.params else {}
    )
    output = output_recommended_recipes(recommendation_dataframe)
    return {"output": output if output else None}
