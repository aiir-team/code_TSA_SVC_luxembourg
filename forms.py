from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Length


class FindTweetsForm(FlaskForm):
    searching = StringField("searching", validators=[DataRequired(), Length(min=5, max=30)])
    submit = SubmitField("Searching")