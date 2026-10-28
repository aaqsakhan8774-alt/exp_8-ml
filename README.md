# exp_8-ml

## Run locally

Install dependencies and train the model:

```powershell
python -m pip install -r requirements.txt
python train.py
```

Run the API after training:

```powershell
python app.py
```

The API listens on port 5001. Send a `POST` request to `/predict` with JSON
fields `sepal_length`, `sepal_width`, `petal_length`, and `petal_width`.
