from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import os
import cv2

if not os.path.exists('best.pt'):
    print("Model file `best.pt` not found in the current directory")
    input("Add the model file and press any key to continue...")

app = FastAPI()

if not os.path.exists('best.pt'):
    raise Exception("Model not found")

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
    }
    
    return JSONResponse(content=response)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)