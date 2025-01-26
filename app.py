import sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QTextEdit,
    QPushButton,
    QMessageBox,
)
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class EmailSender(QWidget):
    """
    This class represents the main window of the application.
    """
    def __init__(self):
        super().__init__()
        self.train_model()
        self.initUI()

    def train_model(self):
        emails = pd.read_csv('emails.csv')

        # Extract features and labels
        X = emails['text']
        y = emails['spam']

        # Convert text data to numerical data
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(X)

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = MultinomialNB()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f'Accuracy: {accuracy}')
        print('Classification Report:')
        print(report)

    def is_spam(self, email_text):
        # Load the dataset
        emails = pd.read_csv('emails.csv')

        # Extract features and labels
        X = emails['text']
        y = emails['spam']

        # Convert text data to numerical data
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(X)

        # Train the model
        model = MultinomialNB()
        model.fit(X, y)

        # Make predictions
        email_text = vectorizer.transform([email_text])
        prediction = model.predict(email_text)

        return prediction[0]
        

    def initUI(self):
        layout = QVBoxLayout()

        self.recipient_label = QLabel("Recipient:")
        self.recipient_input = QLineEdit()
        self.recipient_input.setPlaceholderText("Enter the recipient email address...")
        self.recipient_input.setStyleSheet(
            """
            QLineEdit {
            font-family: Arial, sans-serif;
            font-size: 14px;
            color: #333;
            background-color: #f9f9f9;
            border: 1px solid #ccc;
            padding: 5px;
            }
        """
        )
        layout.addWidget(self.recipient_label)
        layout.addWidget(self.recipient_input)

        self.subject_label = QLabel("Subject:")
        self.subject_input = QLineEdit()
        self.subject_input.setPlaceholderText("Enter the subject of your email...")
        self.subject_input.setStyleSheet(
            """
            QLineEdit {
            font-family: Arial, sans-serif;
            font-size: 14px;
            color: #333;
            background-color: #f9f9f9;
            border: 1px solid #ccc;
            padding: 5px;
            }
        """
        )
        layout.addWidget(self.subject_label)
        layout.addWidget(self.subject_input)

        self.body_label = QLabel("Body:")
        self.body_input = QTextEdit()
        self.body_input.setPlaceholderText("Enter your message here...")
        self.body_input.setFixedHeight(100)
        self.body_input.setStyleSheet(
            """
            QTextEdit {
            font-family: Arial, sans-serif;
            font-size: 14px;
            color: #333;
            background-color: #f9f9f9;
            border: 1px solid #ccc;
            padding: 5px;
            }
        """
        )
        layout.addWidget(self.body_label)
        layout.addWidget(self.body_input)

        self.send_button = QPushButton("Send")
        self.send_button.setStyleSheet(
            """
            QPushButton {
            font-family: Arial, sans-serif;
            font-size: 14px;
            color: #fff;
            width: 10px;
            margin: auto;
            background-color: #007BFF;
            border: 1px solid #007BFF;
            padding: 8px 16px;
            }
            QPushButton:hover {
            background-color: #0056b3;
            border: 1px solid #0056b3;
            }
            QPushButton:pressed {
            background-color: #004380;
            border: 1px solid #004380;
            }
        """)
        self.send_button.clicked.connect(self.send_email)
        layout.addWidget(self.send_button)

        self.setLayout(layout)
        self.setWindowTitle("Email Sender")
        self.setGeometry(100, 100, 800, 600)
        self.show()

    def send_email(self):
        recipient = self.recipient_input.text()
        subject = self.subject_input.text()
        body = self.body_input.toPlainText()

        if not recipient or "@" not in recipient:
            QMessageBox.critical(
                self, "Error", "Please enter a valid recipient email address."
            )
            return

        if not subject:
            QMessageBox.critical(self, "Error", "Please enter a subject.")
            return

        if not body:
            QMessageBox.critical(self, "Error", "Please enter the body content.")
            return

        msg = MIMEMultipart()
        msg["From"] = "your_email@example.com"
        msg["To"] = recipient
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        try:

            # Check if email is spam
            if self.is_spam(body):
                QMessageBox.critical(self, "Error", "The email body is classified as spam.")
                return
            else:
                QMessageBox.information(self, "Success", "Email is not a spam")

            QMessageBox.information(self, "Success", "Email sent successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to send email: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = EmailSender()
    sys.exit(app.exec_())
