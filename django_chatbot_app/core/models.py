from django.db import models
from django.contrib.auth.models import AbstractUser
import uuid

class User(AbstractUser):
    pass

class ChatConversation(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    conversation = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)

    def get_first_user_message(self):
        for message in self.conversation:
            if message['role'] == 'user':
                return message['content']
        return None
    
