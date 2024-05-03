from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import os
import cv2

if not os.path.exists('best.pt'):
    model_url = 'https://drive.afzalsa571.workers.dev/download.aspx?file=j8p8Wc4%2F56r78o%2BJQV1r9o6Zn6%2BgEmEtELsGPWgJao4n4ZWwKheV48f%2BaiITP%2BFW&expiry=0Gy8Z2dE2IL5dLIOvGFXXQ%3D%3D&mac=33998b834c196b4fa47bb3434a26d5d6ef3f28eca949211e5ffcbab3e57f477e'
    os.system(f'wget {model_url} -O best.pt')

app = FastAPI()

model = YOLO('best.pt')

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    filename = file.filename

    with open(filename, "wb") as buffer:
        buffer.write(await file.read())

    results = model.predict(source=filename)
    
    boxes = results[0].boxes.xyxy
    classes = results[0].boxes.cls
    class_names = results[0].names

    
    # draw boxes on image
    img = cv2.imread(filename)
    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])  # Convert coordinates to integers
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, class_names[int(classes[i])], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)    
    
    image_path = f"output_{filename}"
    cv2.imwrite(image_path, img)


    response = {
        "image": image_path,
        "predictions": class_names[int(classes)],
        "class_names": class_names
    }
    
    return JSONResponse(content=response)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)