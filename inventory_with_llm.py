import logging
import re
import os
import discord
from dotenv import load_dotenv
from firebase_admin import credentials, initialize_app, storage, db
from transformers import T5Tokenizer, T5ForConditionalGeneration



# Load pre-trained T5-small model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('fine_tuned_t5_complex')
tokenizer = T5Tokenizer.from_pretrained('fine_tuned_t5_complex')

load_dotenv() #loads the environment variables


def parse_item_string(input_string):
    """
    Parse an input string for item quantity and name.
    Format: "[quantity] item name" or "item name"
    Returns tuple of (quantity, item_name)
    Figured I'd make it a function since it's versatile and might come up later
    """
    # Look for a number at the start followed by whitespace
    quantity_match = re.match(r'^(\d+)\s+(.+)$', input_string)
    
    if quantity_match:
        # If found, use the number and remaining text
        quantity = int(quantity_match.group(1))
        item_name = quantity_match.group(2).strip()
    else:
        # If no number found, default to quantity 1
        quantity = 1
        item_name = input_string.strip()
    
    return quantity, item_name

def case_insensitive_get(inventory, key):
    for item_key in inventory:
        if item_key.lower() == key.lower():
            return item_key, inventory[item_key]  # Return the actual key and the data
    return None, None


def pluralize_phrase(phrase):
    input_string = f"{'pluralize'}: {phrase}"
    inputs = tokenizer(input_string, return_tensors="pt", padding=True, truncation=True, max_length=64)
    outputs = model.generate(inputs['input_ids'])
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def singularize_phrase(phrase):
    input_string = f"{'singularize'}: {phrase}"
    inputs = tokenizer(input_string, return_tensors="pt", padding=True, truncation=True, max_length=64)
    outputs = model.generate(inputs['input_ids'])
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text



def add_item_to_inventory(item_name, quantity_to_add, user_ref):
    try:
        # Retrieve current inventory from the database
        inventory = user_ref.child('inventory').get() or {}
        
        # Get the plural form of the item name
        plural_name = pluralize_phrase(item_name)
        item_name = singularize_phrase(item_name)

        # Try to get the item using either singular or plural form, and agnostic to case
        # Try to get the item (agnostic to case) and retrieve both key and data
        actual_key, item_data = case_insensitive_get(inventory, item_name)
        if not actual_key:
            actual_key, item_data = case_insensitive_get(inventory, plural_name)


        #if there's no item, just add the thing in the quantity        
        if item_data is None:
            print(f"Neither {item_name} nor {plural_name} found in inventory.")
            actual_key = plural_name if quantity_to_add > 1 else item_name
            
            user_ref.child('inventory').child(actual_key).set({
                'quantity': quantity_to_add
            })

            print(f"Successfully added {quantity_to_add} {actual_key}")
            print(f"Current inventory: {user_ref.child('inventory').get()}")
            return actual_key  # Return the actual key for correct messaging
            

        # Calculate new quantity
        new_quantity = item_data['quantity'] + quantity_to_add

        
       # Handle pluralization if quantity exceeds 1 (which it will by logic if the item already exists)
        if new_quantity > 1 and actual_key == item_name:
            # Remove the singular entry and add pluralized version
            user_ref.child('inventory').child(actual_key).delete()
            actual_key = plural_name
            print(f"Converted {item_name} to plural form {actual_key} due to quantity.")

        # Update the item with the new quantity
        user_ref.child('inventory').child(actual_key).update({
            'quantity': new_quantity
        })


        print(f"Successfully added {quantity_to_add} {item_name}")
        print(f"Current inventory: {user_ref.child('inventory').get()}")
        return item_name #item name is returned so message sent in parent functionhas correct plural or singular form
        
    except Exception as e:
        print(f"Error updating inventory: {e}")
        return False




def remove_item_from_inventory(item_name, quantity_to_remove, user_ref):
    try:
        # Retrieve current inventory from the database
        inventory = user_ref.child('inventory').get() or {}

        # Get the plural and singular forms of the item name
        plural_name = pluralize_phrase(item_name)
        singular_name = singularize_phrase(item_name)

        # Try to get the item using either singular or plural form (returns actual key and data)
        actual_key, item_data = case_insensitive_get(inventory, singular_name)
        if not actual_key:
            actual_key, item_data = case_insensitive_get(inventory, plural_name)

        if item_data is None:
            print(f"Neither {singular_name} nor {plural_name} found in inventory.")
            return False

        # Check if enough items are available to remove
        if item_data['quantity'] < quantity_to_remove:
            print(f"Not enough items to remove. Current quantity: {item_data['quantity']}")
            return False

        # Calculate new quantity after removal
        new_quantity = item_data['quantity'] - quantity_to_remove

        if new_quantity == 0:
            user_ref.child('inventory').child(actual_key).delete()
            print(f"Removed all {actual_key}. Item deleted from inventory.")
            return True

        # If only one item is left and the current key is plural, change to singular
        if new_quantity == 1 and actual_key == plural_name:
            # Delete plural entry and create singular entry
            user_ref.child('inventory').child(actual_key).delete()
            user_ref.child('inventory').child(singular_name).set({
                'quantity': new_quantity,
                **{k: v for k, v in item_data.items() if k != 'quantity'}
            })
            print(f"Reduced {plural_name} to a single {singular_name}.")
        else:
            # Update the quantity under the existing key
            user_ref.child('inventory').child(actual_key).update({'quantity': new_quantity})
            print(f"Updated {actual_key}: New quantity is {new_quantity}.")
        #item name is returned so message sent in parent functionhas correct plural or singular form
        if quantity_to_remove == 1: 
            return item_name 
        else:
            return plural_name

    except Exception as e:
        print(f"An error occurred: {e}")
        return False




private_key_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
cred = credentials.Certificate(private_key_path)
initialize_app(cred, {
    'storageBucket': 'dylans-discord-bot.appspot.com',
    'databaseURL': 'https://dylans-discord-bot-default-rtdb.firebaseio.com/'  # Ensure this is correct
    })
bucket = storage.bucket()       #creates the cloud storage bucket





handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)






@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')


@client.event

async def on_message(message):
    if message.author == client.user:  #of course, don't respond to your own messages. thats dumb.
        return
    
    #Called upon for help
    if re.search('^.help', message.content) is not None:
        await message.channel.send("Hi. If you haven't yet, type .initme to init your folder. Then, type .add to add an item to your inventory.")

    #Called upon for inventory
    if re.search('^.inventory', message.content) is not None:
        inventory_ref=db.reference('users').child(str(message.author.id)).child('inventory')
        inventory=inventory_ref.get()
        if inventory is None:
            await message.channel.send("You have no items in your inventory.")
        else:
            inventory_string=""
            for item in inventory:
                inventory_string+=f"{item}: {inventory[item]['quantity']}\n"
            await message.channel.send(inventory_string)


    #Called upon for init
    if re.search('^.initme', message.content) is not None: #this inits a new user for the inventory system
        print("initme called")
        new_username=message.author.name
        new_user_id=message.author.id
        print("got user data")


        try:
            ref = db.reference('users')
            print(f"Reference: {ref}")
        except Exception as e:
            print(f"Error creating reference: {e}")


        try:
            user_ref=ref.child(str(new_user_id)) #refers to the user's node in the database
            print(f"child ref: {user_ref}")
            
        except Exception as e:
            print(f"Error creating child: {e}")


        try:
            user_ref.set({
            'username':new_username,
            'inventory':{}
            })
        except Exception as e:
            print(f"Error setting user data: {e}")

        await message.channel.send(f"Inventory system initialized for {new_username}!")


    #called upon to add an item
    if re.search('^.add', message.content) is not None:  #if .add is called upon, respond
        user_id = message.author.id
        addition_string = message.content.split(".add ", 1)[1].strip()
        item_quantity, item_name = parse_item_string(addition_string)
        print(f"Addition string: {addition_string}")
        print(f"Item quantity: {item_quantity}")
        print(f"Item name: {item_name}")
        try:
            ref = db.reference('users') #goes to users
            print(f"Reference: {ref}")
        except Exception as e:
            print(f"Error creating reference: {e}")


        try:
            user_ref=ref.child(str(user_id)) #refers to the user's node in the database
            print(f"child ref: {user_ref}")
            
        except Exception as e:
            print(f"Error creating child: {e}")


        try:
            item_name=add_item_to_inventory(item_name,item_quantity,user_ref) #add to inventory also returns the name with the correct pluralization, so message can be sent in async parent function

            
        except Exception as e:
            print(f"Error adding item {e}")

        await message.channel.send(f"Successfully added {item_quantity} {item_name}")
        return

    if re.search('^.remove', message.content) is not None:  #This deletes an item, or multiple
        user_id = message.author.id
        deletion_string = message.content.split(".remove ", 1)[1].strip()
        item_quantity, item_name = parse_item_string(deletion_string)
        print(f"Deletion string: {deletion_string}")
        print(f"Item quantity to delete: {item_quantity}")
        print(f"Item name: {item_name}")

        try:
            ref = db.reference('users') #goes to users
            print(f"Reference: {ref}")
        except Exception as e:
            print(f"Error creating reference: {e}")


        try:
            user_ref=ref.child(str(user_id)) #refers to the user's node in the database
            print(f"child ref: {user_ref}")
            
        except Exception as e:
            print(f"Error creating child: {e}")


        try:
            remove_item_from_inventory(item_name,item_quantity,user_ref)

            
        except Exception as e:
            print(f"Error removing item {e}")
        return



















"""     #you can also uplaod music to this bot. I'll keep it here because its not hurting anything even if the focus has switched.
    if re.search('^.upload', message.content) is not None:  #only responds to being called upon
        
        #if there is no attachment to upload
        if not message.attachments:
            return
        ##SONG UPLOAD SYSTEM! This should be reconfigured later to maybe do directly 
        # from discord rather than having to temp uplaod to the workspace so I can host it on a rasberry pi. Not a great bot if it relies on my laptop being online
        for i in message.attachments:
            print(i)
            first_file = i
            print ("HERE???")
            file_name = first_file.filename
            print("check check cehck")
            if re.search('mp3$',file_name) is None:  #only mp3 files get in
                await message.channel.send("Only takes mp3 files at this time")
                return
            print("mp3 status checked")
            if first_file.size > 5000000:
                await message.channel.send("please upload shorter files I'm begging you")
                return
            print("file length checked")
            print("saving file")
            await first_file.save(f'{file_name}')
            #creates blob of file given and puts it in cloud storage bucket. Also has custom metadata
            file_to_upload=file_name
            blob = bucket.blob(file_to_upload)
            userid=message.author.id
            special_metadata={'userid':userid}
            blob.metadata=special_metadata
            blob.upload_from_filename(file_to_upload) 
            blob.make_public() #This gives public access from the URL.
            print("your file url", blob.public_url) #and this prints the link in the console
            await message.channel.send("file saved to database as " + file_name)
            
 """


DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
client.run(DISCORD_TOKEN,log_handler=handler)


















