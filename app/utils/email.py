import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

async def send_password_reset_email(email: str, code: str):
    """Envoie un email de réinitialisation de mot de passe"""
    try:
        # Configuration SMTP
        smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", 587))
        smtp_username = os.getenv("SMTP_USERNAME")
        smtp_password = os.getenv("SMTP_PASSWORD")
        from_email = os.getenv("FROM_EMAIL", smtp_username)

        if not all([smtp_server, smtp_port, smtp_username, smtp_password]):
            print("Configuration SMTP incomplète. Email simulé:")
            print(f"Destinataire: {email}")
            print(f"Code: {code}")
            return

        # Création du message
        subject = "Réinitialisation de votre mot de passe"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Réinitialisation de mot de passe</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #007bff; color: white; padding: 20px; text-align: center; }}
                .content {{ background-color: #f8f9fa; padding: 30px; }}
                .code {{ font-size: 24px; font-weight: bold; color: #007bff; text-align: center; margin: 20px 0; }}
                .footer {{ text-align: center; margin-top: 30px; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Réinitialisation de mot de passe</h1>
                </div>
                <div class="content">
                    <p>Bonjour,</p>
                    <p>Vous avez demandé à réinitialiser votre mot de passe.</p>
                    <p>Votre code de vérification est :</p>
                    <div class="code">{code}</div>
                    <p>Ce code expirera dans 1 heure.</p>
                    <p>Si vous n'avez pas fait cette demande, veuillez ignorer cet email.</p>
                </div>
                <div class="footer">
                    <p>Cet email a été envoyé automatiquement, merci de ne pas y répondre.</p>
                </div>
            </div>
        </body>
        </html>
        """

        text_content = f"""
        Réinitialisation de mot de passe

        Vous avez demandé à réinitialiser votre mot de passe.
        Votre code de vérification est : {code}
        
        Ce code expirera dans 1 heure.
        
        Si vous n'avez pas fait cette demande, veuillez ignorer cet email.
        """

        # Création du message
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = from_email
        message["To"] = email
        
        # Parties texte et HTML
        part1 = MIMEText(text_content, "plain")
        part2 = MIMEText(html_content, "html")
        
        message.attach(part1)
        message.attach(part2)

        # Envoi de l'email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.sendmail(from_email, email, message.as_string())
        
        print(f"Email envoyé à {email}")

    except Exception as e:
        print(f"Erreur lors de l'envoi de l'email: {e}")
        # Fallback: afficher le code dans les logs
        print(f"Code de réinitialisation pour {email}: {code}")