# Payment Risk Scorer

Payment Risk Scorer is a machine learning service that scores synthetic payment transactions for fraud and compliance risk, returning a clear score, a risk level, and human readable reasons for every payment.

[Try the live demo](https://payment-risk-scorer.onrender.com)

[View the code on GitHub](https://github.com/MinuuLakshmi17/Payment-Risk-Scorer)

## Why it matters

Payment teams must spot fraud and compliance risk fast, without blocking good payments. A score alone does not tell a reviewer what to do next. This project pairs every score with reasons in plain words, so a reviewer can check a payment, clear it, or escalate it with confidence. It shows the full flow in honest form: synthetic data, a trained model with measured results, clear explanations, and a live API with a working demo page.

## Key features

* Synthetic transaction generator with a fixed seed, so every run can be repeated exactly
* PyTorch risk model trained on synthetic payment data, with measured results on a held out test set
* Explainable scoring: every result includes a score from 0 to 1, a risk level of low, medium, or high, and reason codes in plain words
* FastAPI service with a health check and a score endpoint, plus a polished demo page that calls the API in a browser
* Light inference path that runs the same trained weights with NumPy only, so the live API stays small and fast without PyTorch
* Full test suite with pytest covering the generator, the model, the reasons, and the API

## How scoring works

* The generator creates transactions with amount, merchant category, country, hour, device type, and velocity
* The encoder turns each transaction into numbers the model can use, with log amount, velocity, hour as sine and cosine, and one hot codes for merchant and device
* The model is a small neural network with two hidden layers and a single output
* The score is the model output as a number from 0 to 1
* Risk level is low below 0.30, medium from 0.30 to below 0.70, and high at 0.70 and above
* Reason codes look at the same transaction a human would check: high amount, very high amount, unusual country, high velocity, night hour, and high risk merchant
* When no strong signal is present, the result says so plainly, instead of inventing a reason

## Live demo

* Try the live demo at the link at the top of this page
* The demo page lets you enter a payment, score it, and see the score, the risk level, and the reasons at once
* The health check lives at the live address with health added at the end
* The live service runs on the free tier, so it may sleep after inactivity and the first request after a pause can be slow

## Tech stack

* Python
* PyTorch for model training
* FastAPI and Uvicorn for the API service
* NumPy for the light inference path
* Pydantic for request validation
* pytest and httpx for tests
* Render for live hosting
* GitHub for source control

## Measured results

All results below are measured on the held out test set, not on the training data. Training used 8000 transactions, 60 epochs, and seed 42, with 5600 transactions for training, 1200 for validation, and 1200 for testing. See models/metrics.json for the full numbers from the latest training run.

* Test set size is 1200 transactions
* Fraud rate in the test set is 0.1892
* Fraud ROC AUC is 0.7350
* Fraud PR AUC is 0.4060
* Fraud accuracy is 0.6808
* Fraud precision is 0.3274
* Fraud recall is 0.6520
* Fraud F1 is 0.4359
* Fraud true positives are 148, true negatives are 669, false positives are 304, and false negatives are 79, at a threshold of 0.5
* Compliance rate in the test set is 0.0375
* Compliance ROC AUC using the same score is 0.9099
* Compliance PR AUC using the same score is 0.4007
* ROC AUC shows how well the score ranks fraud above good payments, with 1.0 as perfect and 0.5 as chance
* PR AUC shows how well the score finds fraud when fraud is rare, with 1.0 as perfect
* Accuracy, precision, recall, and F1 are measured at a threshold of 0.5

## How to run locally

* Install Python 3.10 or newer
* Install the main packages with pip

```
pip install torch fastapi uvicorn httpx numpy pytest pydantic
```

* Train the model from the project folder

```
python train.py
```

* Training makes synthetic data, trains the model, and writes the model, the light weights file, and the metrics file
* Run the API from the project folder

```
python app.py
```

* Open this address in a browser to use the demo page

```
http://localhost:8000
```

* The health check is at GET /health
* The score endpoint is POST /score
* Send amount, merchant category, country, hour, device type, and velocity
* The response has score, risk level, and reasons

## How to deploy on Render

* The repo includes a Render blueprint file named render.yaml
* Render installs only the light requirements file, so the deploy stays small and fast
* The light path uses the same trained weights, stored as plain JSON, so scores match the PyTorch model
* The full requirements file is still used for training, which needs PyTorch
* Create a free Render account and create a new web service from the GitHub repo, then choose the free instance type
* Render will show a live web service URL on the dashboard after deploy
* Open that URL in a browser to see the demo page
* To point the demo page at the live API from another host, set API_BASE_URL at the top of static/index.html to the live web service URL, then redeploy the demo page
* If the demo page is served by the same API service, leave API_BASE_URL empty and it will use the same host
* The free tier may sleep after inactivity, so the first request after a pause can be slow
* The trained model file is not pushed to GitHub, because it is binary. Run train.py once after cloning to make it, or use the light weights file, which is pushed and works without PyTorch

## Tests

* Run the tests from the project folder

```
pytest
```

* The suite covers the generator, the model, the light model, the reasons, and the API
* 23 tests passing, with 0 failed
* All tests should pass before you share the project

## Project layout

* train.py trains the model and writes metrics and weights
* app.py runs the FastAPI service
* export_weights.py converts the trained model to the weights file
* src/generator.py makes synthetic transactions and encodes features
* src/model.py defines the model and the metric code
* src/explain.py defines reason codes, risk levels, and scoring
* src/light_model.py runs the same model with NumPy only, using the weights file
* static/index.html is the demo page
* tests/test_generator.py tests the generator
* tests/test_model.py tests the model
* tests/test_light_model.py tests the light model
* tests/test_explain.py tests the reasons
* tests/test_api.py tests the API
* requirements.txt lists the packages for training and local use
* requirements_light.txt lists only the packages needed to run the API without PyTorch
* render.yaml is the Render blueprint for live hosting
* Dockerfile is an optional container setup for the light runtime
* models/model.pt is the trained model, made by train.py
* models/weights.json holds the trained weights as plain JSON, made by train.py
* models/metrics.json holds the measured metrics, made by train.py
