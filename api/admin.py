from django.contrib import admin
from .models import QueryLog

@admin.register(QueryLog)
class QueryLogAdmin(admin.ModelAdmin):
    list_display = ('question', 'created_at')
    # Make it read-only so it works like a true history log
    readonly_fields = ('question', 'answer', 'sources', 'created_at')