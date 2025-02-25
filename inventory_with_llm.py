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

def get_all_display_names():
    ref = db.reference('users')
    users_snapshot = ref.get()
    
    # Create a dictionary mapping display_names to user_ids
    display_name_to_id = {}
    
    if users_snapshot:
        for user_id, user_data in users_snapshot.items():
            if 'display_name' in user_data:
                display_name_to_id[user_data['display_name']] = user_id
    
    return display_name_to_id



#given the chance to build this entire thing again, I'd probably go full LLM with it, instead of the weird combination of LLM and regex I have right now.
#It probably wouldn't be the most difficult thing in the world to hook this up to a larger gpt model that can just translate natural dm language into inventory actions. I'm sure somebody has done it by now.
#On the bright side, this way is cheaper, the pluralization bot was a good first project, and it feels natural enough for me and my friends.
def parse_natural_language(message_content: str, display_names: dict[str, str]):
    """
    Parse natural language gain/lose messages with multiple items.
    
    Args:
        message_content: The message text to parse
        display_names: Dictionary mapping user_ids to display_names
    
    Returns:
        Tuple of (user_id, list of (quantity, item_name) tuples) or (None, None) if no match
    """

    print("all display names", display_names)
    # Pattern to match a display name followed by an inventory action verb. Can easily add or change what those verbs are
    display_pattern = r"\b(" + "|".join(re.escape(name) for name in display_names) + r")\s+(?:gains|GAINS|loses|LOSES)"

    
    # Check if we have a valid display name match
    name_match = re.search(display_pattern, message_content)
    if not name_match:
        return None, None
    
    # Extract the display name
    display_name = name_match.group(1).strip()
    print("Display name:", display_name)
    
    # Find user_id from display_name. the display names are the keys actually and the ids the values. If anybody who wants to hire me is looking at this i promise I won't do shit this stupid when somebody's paying me.
    user_id = display_names.get(display_name)

    if not user_id:
        return None, None
    print("User ID:", user_id)
    # Get everything after "GAINS"
    gains_text = message_content.split(name_match.group(0), 1)[1].strip()
    print("gains text:", gains_text)

    
    # Define pattern for terminal characters
    # Periods are external if followed by:
    # 1. A lowercase letter
    # 2. A space and then a lowercase letter
    # 3. A space, a capital letter, and then a lowercase letter
    # 4. A space, a capital letter, a space, and then a lowercase letter
    terminal_pattern = r'(?:,\s*(?:and\s+)?|\.(?:\s*[a-z]|\s+[A-Z][a-z]|\s+[A-Z]\s+[a-z])|\s+and\s+|(?:\s+(?:for|as|but|though|although|while|when|if|because|since|after|before|during|from|with|by|to|at|in|on|of|that|which|who|whom|whose|where|how|why|what)\b)|$)'
    
    # Pattern to find quantities and items (allow periods within item names)
    # Periods are internal if not matched by the terminal pattern
    item_pattern = r'(?:(\d+)\s+)?([A-Z][A-Z\s\.]*?(?=' + terminal_pattern + '))'
    # Find all matches
    items = []
    item_matches = re.finditer(item_pattern, gains_text)
    
    for match in item_matches:
        quantity = match.group(1)
        item_name = match.group(2).strip()
        
        # Default quantity to 1 if not specified
        if not quantity:
            quantity = "1"
        
        items.append((int(quantity), item_name))
    
    return user_id, items


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



#I only tuned my model on lowercase data, and it was having trouble with the all caps items. It pluralizes and singularlizes normalized text now, and items are always uppercase. Sorry if you don't like uppercase items. Thank god I do I think they are neat.
def pluralize_phrase(phrase):
    # Normalize input to lowercase
    normalized_phrase = phrase.lower()
    
    # Prepare input string for the model
    input_string = f"{'pluralize'}: {normalized_phrase}"
    inputs = tokenizer(input_string, return_tensors="pt", padding=True, truncation=True, max_length=64)
    
    # Generate output
    outputs = model.generate(inputs['input_ids'])
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Convert output to all caps
    return generated_text.upper()


def singularize_phrase(phrase):
    # Normalize input to lowercase
    normalized_phrase = phrase.lower()
    
    # Prepare input string for the model
    input_string = f"{'singularize'}: {normalized_phrase}"
    inputs = tokenizer(input_string, return_tensors="pt", padding=True, truncation=True, max_length=64)
    
    # Generate output
    outputs = model.generate(inputs['input_ids'])
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Convert output to all caps
    return generated_text.upper()


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
    display_names=get_all_display_names() #makes sure display names dict is up to date before doing anything
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
    if re.search('^.initme', message.content) is not None: #this inits a new user for the inventory system. It adds a display name to the dictionry, so they can be identified by that as well.
        print("initme called")
        display_name = message.content.split(".initme ", 1)[1].strip()
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
            'display_name':display_name,
            'inventory':{}
            })
        except Exception as e:
            print(f"Error setting user data: {e}")

        await message.channel.send(f"Inventory system initialized for {new_username}!")


    #called upon to add an item
    if re.search('^.add', message.content) is not None:  #if .add is called upon, respond. This function is used when the player themselves adds something
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
            await message.channel.send(f"Successfully removed {item_quantity} {item_name}.")


            
        except Exception as e:
            print(f"Error removing item {e}")
        return




    #the following functions are equivalent to the above add and remove, but are controlled by a third party, probably the DM. It's really a matter of preference whether you want to use one or the other or both of these systems.
    # It uses display names and regex as identifiers.
    #I would just hook the entire thing up to a cloud llm for this kind of stuff, but I'm stingy and this way works fine.


    if re.search(r"\b(" + "|".join(re.escape(name) for name in display_names.keys()) + r")\sGAINS|gains", message.content) is not None:
        
        print("dm_add_regex called")
        user_id, items = parse_natural_language(message.content, display_names) #uses the regex function above to produce a clean list of items
        if not user_id or not items:
            return  # No valid gains found
        # Reference to the user in the database
        ref = db.reference('users')
        user_ref = ref.child(str(user_id))
        # Process each item
        response_parts = []
        for quantity, item_name in items:
            # Add to inventory 
            formatted_name = add_item_to_inventory(item_name, quantity, user_ref)
            response_parts.append(f"{quantity} {formatted_name}")
        
        # Create a natural language response
        if len(response_parts) == 1:
            response = f"Successfully added {response_parts[0]}"
        elif len(response_parts) == 2:
            response = f"Successfully added {response_parts[0]} and {response_parts[1]}"
        else:
            last_item = response_parts.pop()
            response = f"Successfully added {', '.join(response_parts)}, and {last_item}"
        
        await message.channel.send(response)


    if re.search(r"\b(" + "|".join(re.escape(name) for name in display_names.keys()) + r")\sLOSES|loses", message.content) is not None:#equivalent function for removing items
        
        print("dm_add_regex called")
        user_id, items = parse_natural_language(message.content, display_names) #uses the regex function above to produce a clean list of items
        if not user_id or not items:
            return  # No valid gains found
        # Reference to the user in the database
        ref = db.reference('users')
        user_ref = ref.child(str(user_id))
        # Process each item
        response_parts = []
        for quantity, item_name in items:
            # Add to inventory 
            formatted_name = remove_item_from_inventory(item_name, quantity, user_ref)
            response_parts.append(f"{quantity} {formatted_name}")
        
        # Create a natural language response
        if len(response_parts) == 1:
            response = f"Successfully removed {response_parts[0]}"
        elif len(response_parts) == 2:
            response = f"Successfully removed {response_parts[0]} and {response_parts[1]}"
        else:
            last_item = response_parts.pop()
            response = f"Successfully removed {', '.join(response_parts)}, and {last_item}"
        
        await message.channel.send(response)



















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


















