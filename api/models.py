from django.db import models

class QueryLog(models.Model):
    question = models.TextField()
    answer = models.TextField()
    sources = models.JSONField()  # Stores the page numbers/sources
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Q: {self.question[:30]}..."