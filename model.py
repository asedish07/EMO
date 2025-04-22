from transformers import BertTokenizerFast, GPT2LMHeadModel, pipeline

def today_feedback(input_text, emotion):
  # model_name = "kykim/gpt3-kor-small_based_on_gpt2"
  # tokenizer = BertTokenizerFast.from_pretrained(model_name)
  # model = GPT2LMHeadModel.from_pretrained(model_name)
  input_text = input_text + f"\n 해당 일기는 {emotion}을 가진 글이야, 응원의 말을 해줘."
  # inputs = tokenizer.encode(input_text, return_tensors="pt")
  # outputs = model.generate(inputs, max_length=150)
  
  # for i, sample_output in enumerate(outputs):
  #   print(">> Generated text {}\n\n{}".format(i+1, tokenizer.decode(sample_output.tolist())))
  #   print('\n---')
  pipe = pipeline("text-generation", model="mykor/gpt2-ko", max_length=150, truncation=True)
  print(pipe(input_text))

input_text = "할 일들이 너무 많다. 머리가 터질 것 같다. 집을 가고 싶다. 할 일들이 너무 많다. 머리가 터질 것 같다. 집을 가고 싶다. 할 일들이 너무 많다. 머리가 터질 것 같다. 집을 가고 싶다. 할 일들이 너무 많다. 머리가 터질 것 같다. 집을 가고 싶다.할 일들이 너무 많다. 머리가 터질 것 같다. 집을 가고 싶다. 할 일들이 너무 많다. 머리가 터질 것 같다. 집을 가고 싶다.할 일들이 너무 많다."
today_feedback(input_text, "슬픔")

# tokenizer = AutoTokenizer.from_pretrained("SEOKDONG/llama3.0_korean_v1.0_sft")
# model = AutoModel.from_pretrained("SEOKDONG/llama3.0_korean_v1.0_sft")

# input_text =  """ 「국민건강보험법」제44조, 「국민건강보험법 시행령」제19조,「약관의 규제에 관한 법률」제5조, 「상법」제54조 참조 판단 해줘""" + " 답변:"
# inputs = tokenizer(input_text, return_tensors="pt")
# with torch.no_grad():
#   outputs = model.generate(**inputs, max_length=1024,  temperature=0.5, do_sample=True, repetition_penalty=1.15)

# result = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(result)