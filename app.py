from flask import Flask, flash, request, redirect, url_for, render_template
import os
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.models import load_model

app = Flask(__name__,static_folder='static')

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Load the TensorFlow model
if not os.path.exists('./Model.h5'):
    raise ValueError("Model not found")
model = load_model('Model.h5')

# Define class names
class_names = ['Bacterial_spot',
               'Early_blight',
               'Late_blight',
               'Leaf_Mold',
               'Septorial_leaf_spot',
               'Spider_mites Two-spotted_spider_mite',
               'Tomato_Yellow_Leaf_Curl_Virus',
               'Tomato_mosaic_virus',
               'Healthy']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    img = Image.open(image_path).convert('RGB').resize((256, 256))
    img_array = np.array(img)
    img_batch = tf.expand_dims(img_array, axis=0)
    pred = model.predict(img_batch)
    pred_class = class_names[np.argmax(pred[0])]
    pred_conf = round(100 * np.max(pred[0]), 2)
    return pred_class, pred_conf

@app.route('/')
def home():
    return render_template('index.html')

disease_details = {
    'Bacterial_spot': {
        'About' : '''Bacterial spot is a common and destructive disease that affects tomato plants. 
                    It is caused by the bacterium Xanthomonas campestris pv. vesicatoria. 
                    This disease thrives in warm,humid conditions and can cause significant yield loss if left unchecked.
                    Bacterial spot can cause significant damage to tomato plants if left unmanaged, leading to reduced yields and quality.
                    By implementing proper cultural practices, using resistant varieties, and applying appropriate chemical controls, growers can effectively manage bacterial spot and minimize its impact on tomato crops.''',
        'Symptoms':'''Bacterial spot typically appears as small, water-soaked lesions on the leaves, which later develop into dark brown or black spots with a yellow halo.
                        The spots can coalesce, leading to extensive damage and defoliation.
                        In severe cases, the infection can affect stems, fruit, and even the plant's vascular system, causing wilting and stunting.''',
        'Environmental Conditions': '''Bacterial spot thrives in warm, humid conditions, with temperatures ranging from 75°F to 85°F (24°C to 29°C) being optimal for disease development.
                                        Rainfall, overhead irrigation, and high humidity facilitate the spread of the bacteria.
                                        The disease can also be transmitted through contaminated seed, plant debris, or by mechanical means.''',
        'Disease Spread' : '''Bacterial spot spreads through water splash, wind, or physical contact with infected plant material.
                            The bacterium can enter the plant through natural openings or wounds, such as those caused by insect feeding or pruning.
                            Once inside the plant, the bacterium multiplies and spreads, leading to the development of symptoms.''',
        'Management' : '''Management of bacterial spot involves a combination of cultural practices, such as crop rotation, sanitation, and planting disease-free seedlings.
                        Avoiding overhead irrigation and watering at the base of plants can help reduce moisture levels and minimize disease spread.
                        Copper-based fungicides or bactericides may be applied preventatively or curatively to control bacterial spot, although resistance to these chemicals can develop over time.''',
        'Resistant Varieties' : '''Some tomato varieties exhibit resistance to bacterial spot, which can be a valuable tool in managing the disease.
                                    Resistant varieties are less susceptible to infection and can help reduce the need for chemical control measures.''',
        'Early Detection and Control' : '''Early detection of bacterial spot is important for effective management.
                                            Regularly inspecting plants for symptoms and promptly removing and destroying infected plant parts can help prevent the spread of the disease.
                                            Infected plants should be carefully disposed of to prevent further contamination of the garden.'''
        },
    'Early_blight': {
        'About' : '''Early blight, caused by the fungus Alternaria solani, is a common fungal disease that affects tomato plants.
                    Early blight can cause significant damage to tomato plants if left unmanaged, leading to reduced yields and quality.
                    By implementing proper cultural practices, using fungicides as needed, and selecting resistant varieties, growers can effectively manage early blight and minimize its impact on tomato crops.''',
        'Symptoms' : ''' Early blight typically manifests as circular or irregularly shaped brown or black spots on the lower leaves of tomato plants.
                        These spots may have concentric rings or a target-like appearance.
                        As the disease progresses, the spots may enlarge and coalesce, causing the affected leaves to yellow, wither, and eventually die.
                         blight can also affect stems and fruit, causing dark lesions and decay.''',
        'Environmental Conditions'  : ''' Early blight thrives in warm, humid conditions, but can also develop during periods of cool, wet weather.
                                            The fungus overwinters in infected plant debris and soil, making crop rotation an important management strategy.''',
        'Disease Spread'    : '''Early blight spreads through spores produced on infected plant tissue. 
                                     spores can be transmitted by splashing water, wind, or through contact with contaminated plant material.
                                    The disease can also be introduced into the garden through infected transplants or contaminated tools.''',
        'Management' : '''Management of early blight typically involves a combination of cultural practices, fungicides, and resistant varieties.
                        Practices such as crop rotation, mulching to reduce soil splash, proper spacing to improve air circulation, and removal of infected plant debris can help reduce the spread of the disease. 
                        Fungicides labeled for early blight control can be applied preventatively or curatively, depending on the severity of the outbreak. 
                        Additionally, planting resistant tomato varieties can help minimize the impact of the disease.''',
        'Early Detection and Control' : '''Early detection of early blight is important for effective control.
                                            Regularly inspecting plants for symptoms and promptly removing and destroying infected plant parts can help prevent the spread of the disease. 
                                            Fungicides may also be applied preventatively to protect healthy foliage, especially during periods of favorable weather for disease development.'''
        },
    'Late_blight': {
        'About'  : '''Late blight, caused by the fungus-like oomycete pathogen Phytophthora infestans,
                    is a devastating disease that affects tomato plants (as well as potatoes and other members of the Solanaceae family).
                    Late blight is highly contagious and can spread rapidly, especially under cool, wet conditions.
                    Spores produced on infected plants can be carried by wind or water to nearby plants, leading to widespread outbreaks''',
        'Disease Spread': '''Late blight is highly contagious and can spread rapidly, especially under cool, wet conditions.
                            Spores produced on infected plants can be carried by wind or water to nearby plants, leading to widespread outbreaks.''',
        'Environmental Conditions': '''The disease thrives in cool, moist environments, with temperatures ranging from 50°F to 80°F (10°C to 27°C) being optimal for its growth and spread.
                                Rainy or humid weather facilitates the development and spread of late blight.''',
        'Management': '''Management strategies for late blight typically include a combination of cultural practices, fungicides, and resistant varieties.
                Practices such as crop rotation, proper sanitation (removal of infected plant debris), and spacing plants to improve air circulation can help reduce the risk of infection.
                Fungicides may be used preventatively or curatively, but resistance to fungicides can develop, so rotation of different types of fungicides is recommended.''',
        'Resistant Varieties': '''Some tomato varieties have been bred to exhibit resistance to late blight.
                        While resistant varieties can significantly reduce the impact of the disease, they may still become infected under severe pressure, so combining resistant varieties with other management practices is often advisable.''',
        'Early Detection and Control': '''Early detection of late blight is crucial for effective control.
                                Regularly inspecting plants for symptoms and promptly removing and destroying infected plant parts can help prevent the spread of the disease.
                                Additionally, timely application of fungicides when environmental conditions are conducive to disease development can help manage outbreaks.'''
    },
    'Leaf_Mold' : {
        'About' : '''Leaf mold is another common fungal disease that affects tomato plants, caused by the pathogen Fulvia fulva (formerly known as Cladosporium fulvum). ''',
        'Symptoms' : '''Leaf mold typically first appears as yellowish or pale green spots on the upper surface of tomato leaves.
                        As the disease progresses, these spots develop into characteristic fuzzy, olive-green to brown patches on the undersides of leaves.
                        Infected leaves may also curl upward, and in severe cases, defoliation can occur.''',
        'Environmental Conditions': '''Leaf mold thrives in warm, humid conditions.
                                Unlike some other tomato diseases like late blight, leaf mold tends to be more prevalent in high humidity rather than in cool, wet conditions.
                                Greenhouse environments with poor air circulation can be particularly conducive to the development of leaf mold.''',
        'Disease Spread': '''Leaf mold is spread primarily through spores produced on infected plant tissue.
                    These spores can be transmitted by splashing water, wind, or through contact with contaminated plant material.
                    The disease can overwinter on infected debris and reintroduce itself in the following growing season.''',
        'Management': '''Management of leaf mold often involves a combination of cultural practices and fungicide applications.
                Practices such as maintaining adequate spacing between plants to improve air circulation, watering at the base of plants to avoid wetting foliage, and removing infected leaves can help reduce the spread of the disease.
                Additionally, applying fungicides labeled for leaf mold control can be effective, especially in areas where the disease is prevalent.''',
        'Resistant Varieties': '''Some tomato varieties exhibit resistance to leaf mold. 
                        These resistant varieties can be a valuable tool in managing the disease, as they are less susceptible to infection and can help reduce the need for fungicide applications.''',
        'Early Detection and Control': '''Early detection of leaf mold is important for effective control.
                                Regularly inspecting plants for symptoms and promptly removing and destroying infected plant parts can help prevent the spread of the disease.
                                Fungicides may also be applied preventatively or curatively, depending on the severity of the outbreak.'''
    },
    'Septorial_leaf_spot' : {
        'About' : '''Septoria leaf spot, caused by the fungus Septoria lycopersici, is a common fungal disease affecting tomato plants.
                    Septoria leaf spot can cause significant damage to tomato plants if left unmanaged, leading to reduced yields and quality. 
                    By implementing proper cultural practices, using fungicides as needed, and selecting resistant varieties, growers can effectively manage Septoria leaf spot and minimize its impact on tomato crops.''',
        'Symptoms': '''Septoria leaf spot typically begins as small, water-soaked spots on the lower leaves of tomato plants.
                        These spots gradually enlarge and develop into characteristic circular lesions with a dark brown margin and a tan to gray center.
                        As the disease progresses, the lesions may merge, causing extensive defoliation and reduced fruit production. 
                        Unlike some other tomato diseases, Septoria leaf spot primarily affects the lower leaves of the plant before moving upward.''',
        'Environmental Conditions': '''Septoria leaf spot thrives in warm, humid conditions, with temperatures ranging from 68°F to 77°F (20°C to 25°C) being optimal for disease development.
                                The fungus overwinters in infected plant debris and soil, making crop rotation an important management strategy.
                                The disease is favored by overhead irrigation and prolonged leaf wetness, so providing adequate air circulation and avoiding wetting foliage can help reduce its spread.''',
        'Disease Spread': '''Septoria leaf spot spreads through spores produced on infected plant tissue. 
                    These spores can be transmitted by splashing water, wind, or through contact with contaminated plant material.
                    The disease can also be introduced into the garden through infected transplants or contaminated tools.''',
        'Management': '''Management of Septoria leaf spot typically involves a combination of cultural practices and fungicide applications.
                Practices such as crop rotation, mulching to reduce soil splash, proper spacing to improve air circulation, and removal of infected plant debris can help reduce the spread of the disease. 
                Fungicides labeled for Septoria leaf spot control can be applied preventatively or curatively, depending on the severity of the outbreak.''',
        'Resistant Varieties': '''Some tomato varieties exhibit partial resistance to Septoria leaf spot, which can help reduce the severity of the disease.
                        While resistant varieties may still become infected, they tend to show fewer symptoms and experience slower disease progression.''',
        'Early Detection and Control': '''Early detection of Septoria leaf spot is important for effective control. 
                                    Regularly inspecting plants for symptoms and promptly removing and destroying infected plant parts can help prevent the spread of the disease.
                                    Fungicides may also be applied preventatively to protect healthy foliage, especially during periods of favorable weather for disease development.'''
    },
    'Spider_mites Two-spotted_spider_mite' : {
        'About' : '''Spider mites, including the two-spotted spider mite (Tetranychus urticae), are common pests that can infest tomato plants. ''',
        'Identification': '''Spider mites are tiny arachnids that feed on the undersides of tomato leaves.
                            The two-spotted spider mite, in particular, is named for the two dark spots visible on its body.
                            These pests are barely visible to the naked eye, appearing as tiny specks moving on the plant. Damage caused by spider mites includes stippling (tiny yellow or white spots) on the upper leaf surface,
                            webbing on the undersides of leaves, and in severe infestations, leaf discoloration, and defoliation.''',
        'Environmental Conditions': '''Spider mites thrive in warm, dry conditions, although they can be problematic in greenhouses as well.
                                        High temperatures and low humidity create ideal conditions for their rapid reproduction.
                                        Dusty conditions can also exacerbate spider mite infestations.''',
        'Damage': '''Spider mites feed on the plant by piercing the leaf tissue and sucking out the sap.
                    This feeding causes cellular damage, leading to the characteristic stippling on the leaves.
                    Severe infestations can weaken the plant, reduce fruit production, and even cause plant death if left untreated.''',
        'Lifecycle': '''Spider mites reproduce rapidly, with females laying hundreds of eggs over their lifespan.
                        Under favorable conditions, populations can explode within a short period, leading to widespread damage.
                        Their short lifecycle allows for multiple generations to occur within a single growing season.''',
        'Management': '''Managing spider mites on tomato plants often involves a combination of cultural, mechanical, and chemical methods. 
                        Cultural method include, Regularly inspecting plants for signs of infestation, removing and destroying heavily infested plant parts, and maintaining good garden hygiene by removing weeds and debris that can harbor mites.
                        Mechanical method includes, Spraying plants with a strong stream of water can help dislodge and reduce spider mite populations. 
                        This method is most effective when done early in the infestation.
                        Chemical method includes, In severe infestations, insecticidal soaps, horticultural oils, or miticides labeled for spider mite control can be used. 
                        Care should be taken to follow label instructions and avoid harming beneficial insects.''',
        'Prevention': '''Preventing spider mite infestations involves maintaining a healthy garden environment, 
                        including adequate watering to avoid drought stress, promoting natural predators such as ladybugs and predatory mites, 
                        and monitoring plants regularly for early signs of infestation.'''
    },
    'Tomato_Yellow_Leaf_Curl_Virus' : {
        'About' : 'Tomato Yellow Leaf Curl Virus (TYLCV) is a devastating viral disease that affects tomato plants, particularly in warm and tropical regions.',
        'Symptoms': ''' TYLCV typically manifests in the symptoms like Yellowing and curling of the leaves, especially on the younger foliage,
                        Stunted growth of the plant.Reduced fruit production and quality,
                        Leaf thickening and development of a leathery texture.Eventually, the entire plant may become stunted and die.''',
        'Transmission': '''TYLCV is primarily transmitted by the silverleaf whitefly (Bemisia tabaci). 
                            These tiny insects feed on the sap of infected plants and then transfer the virus to healthy plants as they feed.''',
        'Viral Characteristics' : '''TYLCV belongs to the family Geminiviridae and is classified as a begomovirus. 
                                    It has a single-stranded DNA genome and is transmitted in a persistent, circulative manner by the whitefly vector.''',
        'Host Range' : '''TYLCV infects plants in the Solanaceae family, with tomatoes being the most susceptible. 
                        However, it can also affect other important crops like peppers, potatoes, and eggplants.''',
        'Management':'''Implementing proper sanitation measures, including the removal and destruction of infected plants, can help reduce the spread of the virus,
                        Managing whitefly populations through the use of insecticides or biological control methods can help mitigate the spread of the virus,
                        Planting resistant tomato varieties can offer some level of protection against TYLCV. 
                        Several tomato cultivars have been bred to exhibit resistance to the virus,
                        Rotating tomato crops with non-host plants can help break the disease cycle and reduce the buildup of viral inoculum in the soil,
                        Regular scouting for symptoms and early detection of the virus can aid in implementing timely management strategies.''',
        'Economic Impact': '''TYLCV can cause significant economic losses in tomato production areas due to reduced yields, increased production costs associated with disease management, and potential trade restrictions on infected produce.'''
    },
    'Tomato_mosaic_virus' :{
        'About' : '''Tomato mosaic virus (ToMV) is a common viral disease that affects tomato plants worldwide. 
                    Understanding its characteristics can help farmers effectively manage and mitigate its impact.''',
        'Symptoms': '''Tomato mosaic virus (ToMV) inflicts noticeable symptoms on tomato plants, including the development of mosaic patterns on leaves characterized by alternating light and dark green areas, resulting in a mottled appearance. 
                        Additionally, infected leaves may exhibit distortion, curling, and puckering. 
                        These symptoms often lead to stunted growth, particularly in younger plants, and can adversely affect fruit yield and quality, causing deformities or discoloration in the fruit.''',
        'Transmission': '''ToMV spreads through multiple channels, making it highly contagious. 
                            Mechanical transmission occurs when the virus is transferred via contaminated tools, hands, or clothing during routine cultivation practices like pruning or harvesting. 
                            Insects, notably aphids, also act as vectors, transmitting the virus from infected to healthy plants. 
                            Furthermore, ToMV can persist in plant debris, soil, and on surfaces for extended periods, facilitating its spread through indirect contact.''',
        'Viral Characteristics' : '''Belonging to the Tobamovirus genus, ToMV possesses a single-stranded RNA genome. 
                                    Its exceptional stability enables it to persist in various environments, contributing to its widespread prevalence. 
                                    The virus's genetic makeup and transmission mechanisms underscore the need for comprehensive management strategies to mitigate its impact on tomato crops.''',
        'Management': '''Effective management of ToMV entails a multifaceted approach. 
                        Maintaining strict sanitation practices is crucial to minimize the risk of virus transmission, including regular disinfection of tools, equipment, and greenhouse surfaces. 
                        Controlling populations of insect vectors, such as aphids, through insecticides or biological control methods helps reduce the spread of the virus. Additionally, planting resistant tomato cultivars and using certified disease-free seeds, coupled with seed treatment methods, can mitigate the risk of infection. 
                        Employing crop rotation with non-host plants and vigilant monitoring for symptoms enable early detection and removal of infected plants, preventing further spread.''',
        'Economic Impact': '''ToMV poses significant economic challenges to tomato growers, as it can lead to substantial losses. 
                                Reduced yields and diminished fruit quality directly impact profitability, while the costs associated with disease management, including labor, materials, and potential crop losses, further exacerbate the financial burden. 
                                Consequently, implementing proactive measures to prevent and control ToMV is essential for safeguarding the economic viability of tomato production operations.'''
    },
    'Healthy' : {
        'About Prediction' : ''' We are happy to says that the system predicted the image that you provided is  healty tomato leaf. ''',
        'Planting Conditions':'''Aim for 6-8 hours of direct sunlight daily. 
                                Choose a well-lit area in your field.
                                Ensure well-drained soil, ideally sandy loam or silt loam with a pH of 6.0-7.0.
                                Avoid heavy clay or very acidic soil.Tomatoes prefer warm weather, with daytime temperatures between 23-29°C (73-84°F) and nighttime temperatures above 15°C (59°F).''',
        'Planting and Care':'''Decide between starting with seeds in a nursery or using pre-grown seedlings.
                                Seeding times will vary based on your climate.When transplanting seedlings, maintain proper spacing as recommended for your specific tomato variety.
                                This allows for good air circulation and fruit development.Provide consistent watering, avoiding oversaturation.
                                Deep watering every few days is better than frequent shallow watering. 
                                Adjust watering based on weather conditions and monitor soil moisture.
                                Use stakes or cages as the plants grow taller to prevent branches from breaking under the weight of the tomatoes.
                                Regularly remove suckers (shoots between the main stem and branches) to improve fruit production and airflow.''',
        'Harvesting':'''Harvest tomatoes when they reach their mature color (red, yellow, orange, etc.) and soften slightly when pressed gently.
                        Carefully pick tomatoes by holding the stem and twisting the fruit to detach it from the vine. Avoid damaging the vine or remaining fruits.''',
        'Additional Tips':'''Practice crop rotation to prevent soilborne diseases. 
                            Don't plant tomatoes in the same location year after year.
                            Apply mulch around the base of the plants to retain moisture, suppress weeds, and regulate soil temperature.
                            Be aware of common tomato pests and diseases in your area.
                            Monitor your plants regularly and take preventative measures or use organic or approved control methods if necessary.'''
               
    }
    
}


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        pred_class, pred_conf = predict_image(file_path)
        flash('Image successfully uploaded and displayed below')
        return render_template('index.html', filename=filename, pred_class=pred_class, pred_conf=pred_conf,disease_details=disease_details)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run(debug=True)
