from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

name = "ali"
role = "web developer"
hobbies = "gamming and photography"

prompt = f"""Write a professional biography.
Name: {name}
Role: {role}
Hobbies: {hobbies}
Write a short professional bio in 2-3 sentences.
Biography:"""

print("Generating bio...")
outputs = generator(
    prompt,
    max_new_tokens=100,  
    do_sample=True,      
    temperature=0.7,     
    top_k=50,            
    top_p=0.9,
    return_full_text=False
)

result_text = outputs[0]['generated_text']

print(result_text)