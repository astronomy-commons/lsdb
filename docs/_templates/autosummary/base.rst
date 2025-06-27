
{%- if objname.split('.')[-1] == objname %}
{{ objname | escape | underline }}
{%- else %}
{{ objname.split('.')[-1] | escape | underline }}
{%- endif %}

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}