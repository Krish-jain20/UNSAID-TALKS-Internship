import re

# Store credentials in a dictionary
vault = {}

# Function to check if password is strong
def is_strong_password(pwd):
    checks = [
        lambda s: any(x.islower() for x in s),
        lambda s: any(x.isupper() for x in s),
        lambda s: any(x.isdigit() for x in s),
        lambda s: any(x in "!@#$%^&*()-_=+[]{}|;:',.<>?/" for x in s),
        lambda s: len(s) > 8
    ]
    return all(check(pwd) for check in checks)

# Classify password strength
def classify_password(pwd):
    score = 0
    if len(pwd) >= 6:
        score += 1
    if any(c.islower() for c in pwd):
        score += 1
    if any(c.isupper() for c in pwd):
        score += 1
    if any(c.isdigit() for c in pwd):
        score += 1
    if any(c in "!@#$%^&*()-_=+[]{}|;:',.<>?/" for c in pwd):
        score += 1

    if score <= 2:
        return "Weak"
    elif score == 3 or score == 4:
        return "Medium"
    else:
        return "Strong"

# Add new credential
def add_credential():
    username = input("Enter username: ")
    password = input("Enter password: ")
    vault[username] = password
    print("✅ Credential added successfully.")

# View all credentials
def view_credentials():
    if not vault:
        print("🔒 Vault is empty.")
    else:
        for user, pwd in vault.items():
            print(f"Username: {user}, Password: {pwd}")

# Delete a credential
def delete_credential():
    user = input("Enter the username to delete: ")
    if user in vault:
        del vault[user]
        print("🗑️ Credential deleted.")
    else:
        print("❌ Username not found.")

# Analyze password strength
def analyze_strength():
    passwords = list(vault.values())
    if not passwords:
        print("🔐 No passwords to analyze.")
        return

    strong_passwords = list(filter(is_strong_password, passwords))
    print("\n🔍 Password Strength Analysis:")
    for user, pwd in vault.items():
        strength = classify_password(pwd)
        print(f"Username: {user} → Password: {pwd} → Strength: {strength}")
    print(f"\n💪 Strong Passwords ({len(strong_passwords)}): {strong_passwords}")

# Main menu loop
def main():
    while True:
        print("\n===== Password Vault Menu =====")
        print("1. Add New Credential")
        print("2. View Credentials")
        print("3. Delete a Credential")
        print("4. Analyze Password Strength")
        print("5. Exit")

        choice = input("Enter your choice (1-5): ")

        if choice == '1':
            add_credential()
        elif choice == '2':
            view_credentials()
        elif choice == '3':
            delete_credential()
        elif choice == '4':
            analyze_strength()
        elif choice == '5':
            print("🔚 Exiting Password Vault. Goodbye!")
            break
        else:
            print("⚠️ Invalid choice. Please try again.")

# Run the program
main()
