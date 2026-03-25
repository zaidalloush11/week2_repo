from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

generator = pipeline('text-generation', model="Qwen/Qwen2.5-0.5B-Instruct")

@app.route('/', methods=['GET', 'POST'])
def home():
    bio = ""
    
    if request.method == 'POST':
        
        name = request.form['name']
        role = request.form['role']
        hobbies = request.form['hobbies']
        
       
        prompt = f"Name: {name}\nRole: {role}\nHobbies: {hobbies}\nWrite a short professional bio:"

      
        output = generator(prompt, max_new_tokens=60, temperature=0.3, return_full_text=False)
        
      
        bio = output[0]['generated_text'].strip()

    return render_template('index.html', generated_bio=bio)

if __name__ == '__main__':
    app.run(debug=True)