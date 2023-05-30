
from utils import *


epsilon_upper = 0 


model = Pix2StructForConditionalGeneration.from_pretrained("google/matcha-chartqa").to(0)
target = (np.array(Image.open("657.png"))[:, :, :3].astype(np.float32)) / 255.0

target_answer = "25"
question = "What's the value of the yellow bar in New York?"

inputs = processor(images=target, text=question, return_tensors="pt").to(0)
prediction = processor.decode(model.generate(**inputs, max_new_tokens=512)[0], skip_special_tokens=True)

print(prediction)

final = boundary_attack(model, eval_model, target, question, target_answer, 1000)


