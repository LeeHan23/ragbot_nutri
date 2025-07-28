import bcrypt

# List the plain-text passwords you want to hash
passwords_to_hash = ['abc', 'def']

print("Please copy the hashed passwords below into your config.yaml file:")
print("----------------------------------------------------------------")

for password in passwords_to_hash:
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password_bytes, salt)
    # Decode for YAML compatibility
    print(f"- {hashed_password.decode('utf-8')}")

print("----------------------------------------------------------------")