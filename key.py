from cryptography.fernet import Fernet

# Generate a Fernet key
key = Fernet.generate_key()

# Save the key to a file named "secret.key"
with open("secret.key", "wb") as key_file:
    key_file.write(key)

print("Fernet key generated and saved to secret.key.")


# git remote add origin https://github.com/jadhav045/Road-Sign-Detection-System.git
