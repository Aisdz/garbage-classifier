# Smart Waste Detection System

![Accuracy](https://img.shields.io/badge/accuracy-92.7%25-brightgreen)
![Python](https://img.shields.io/badge/python-3.10-blue)
![Framework](https://img.shields.io/badge/framework-Flask-black)

# Garbage classifier 🗑️

EfficientNetB0 transfer learning model trained on the 
[Garbage Classification v2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2) 
dataset (10 classes, ~10k images).

## Results
| Model | Test Accuracy |
|-------|--------------|
| Logistic Regression | ~35% |
| SVM (RBF) | ~57% |
| Random Forest | ~58% |
| **EfficientNetB0** | **~93%** |

## Classes
battery, biological, cardboard, clothes, glass, 
metal, paper, plastic, shoes, trash

## Training
See the notebook for the full pipeline:
data exploration → train/val/test split → 
classical ML baselines → EfficientNetB0 transfer learning


## How to run the web app
1. Download the model: 
Place `garbage_efficientnet_final.keras` in the project folder
2. Install dependencies:
   pip install -r requirements.txt,
   template/index.html
4. Run:
   python app.py
5. Open: http://localhost:3000
 
 <img width="1676" height="956" alt="Screenshot 2026-05-11 at 23 13 14" src="https://github.com/user-attachments/assets/e836a165-c429-4a8a-a8f4-d02f4023f84a" />
