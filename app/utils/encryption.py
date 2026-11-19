"""End-to-End Encryption and In-Memory Storage for Secure Chat"""
from cryptography.fernet import Fernet
import time

class ChatEncryption:
    def __init__(self):
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)

    def encrypt(self, message: str) -> str:
        return self.cipher.encrypt(message.encode("utf-8")).decode("utf-8")

    def decrypt(self, encrypted_message: str) -> str:
        return self.cipher.decrypt(encrypted_message.encode("utf-8")).decode("utf-8")

    def get_key_preview(self) -> str:
        return self.key.decode("utf-8")[:16] + "..."

class ChatStorage:
    """Simple in-memory storage to hold encrypted messages between users."""
    def __init__(self):
        self.messages = []

    def add_message(self, sender: str, recipient: str, encrypted_payload: str):
        self.messages.append({
            "id": len(self.messages) + 1,
            "timestamp": time.strftime("%H:%M:%S"),
            "sender": sender,
            "recipient": recipient,
            "payload": encrypted_payload
        })

    def get_messages_for(self, user: str):
        """Get all encrypted messages addressed to a specific user"""
        return [m for m in self.messages if m["recipient"] == user]

    def clear(self):
        self.messages = []