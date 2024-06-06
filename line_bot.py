from flask import Flask, request, abort
import requests
import json
import joblib
import numpy as np
from io import BytesIO
from PIL import Image
from skimage.transform import resize

app = Flask(__name__)
LINE_CHANNEL_ACCESS_TOKEN = 'qHOkYDOx3Dk5SwMOaxTC8fAl/CBgaLHueQRp1fQ5zmY2U8mKz77IfMO11myhWMCQh1HGwbkt1LBnkUsLRBmda4/At99YfKHkajLjS6Fp4NOL8l0PIYvOXEcoTaKkxUcSD4zJcaRNZlipjErbbIIQdgdB04t89/1O/w1cDnyilFU='


def reply_message(reply_token, message):
    url = "https://api.line.me/v2/bot/message/reply"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"
    }
    payload = {
        "replyToken": reply_token,
        "messages": [{
            "type": "text",
            "text": message
        }]
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.status_code, response.text


@app.route("/webhook", methods=['POST'])
def webhook():
    if request.method == 'POST':
        body = request.json
        for event in body['events']:
            if event['type'] == 'message' and event['message']['type'] == 'text':
                reply_token = event['replyToken']
                user_message = event['message']['text']
                reply_message(reply_token, f"You said: {user_message}")

            elif event['message']['type'] == 'image':
                reply_token = event['replyToken']
                message_id = event['message']['id']

                # ดาวน์โหลดภาพจากไลน์
                image_url = f"https://api-data.line.me/v2/bot/message/{message_id}/content"
                headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}
                image_response = requests.get(image_url, headers=headers, stream=True)

                if image_response.status_code == 200:
                    try:
                        image = Image.open(BytesIO(image_response.content))
                        img_resized = resize(np.array(image), (64, 64), anti_aliasing=True).flatten().reshape(1, -1)
                        prediction = loaded_knn.predict(img_resized)
                        predicted_label = "อ้อยเป็นโรค" if prediction[0] == 1 else "อ้อยไม่เป็นโรค"

                        reply_message(reply_token, predicted_label)
                    except Exception as e:
                        reply_message(reply_token, "ไม่สามารถประมวลผลภาพได้")
                        print(f"Error processing image: {e}")
                else:
                    reply_message(reply_token, "ไม่สามารถดาวน์โหลดภาพได้")

        return 'OK'
    else:
        abort(400)


if __name__ == "__main__":
    # โหลดโมเดล
    model_path = 'knn_model.joblib'
    loaded_knn = joblib.load(model_path)
    app.run()
