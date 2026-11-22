# Deploy the dashboard

This repository includes the saved models and NSL-KDD data in verified asset parts.
The dashboard restores them automatically at startup. Host the dashboard;
do not run `run_all.py` during deployment. It retrains the models and takes much longer.

## Streamlit Community Cloud

1. Connect this GitHub repository to Streamlit Community Cloud.
2. Set the entrypoint to `dashboard/app.py` and Python to 3.12.
3. Create a free Neon PostgreSQL project. In Neon's **Connect** dialog, copy its
   pooled connection string and replace the password placeholder with your database
   password. Use the string with `sslmode=require`.
4. In Streamlit's **Advanced settings > Secrets**, add:

   ```toml
   DATABASE_URL = "postgresql://..."
   ADMIN_PASSWORD = "choose-a-long-unique-password"
   ```

   Keep these values out of GitHub and chat. Streamlit exposes top-level secrets as
   environment variables to the app.
5. Deploy. The app creates two small database tables automatically on first start.
   It seeds the Q-learning policy from `models/nslkdd/q_table.json` only if the
   database has no policy yet.

The dashboard is public if you choose public sharing. Its Human Feedback page
requires `ADMIN_PASSWORD` to view or submit feedback. Model predictions and the
current response policy are visible on other pages. A shared password is suitable
for a demonstration, not a production analyst authentication system.

If `DATABASE_URL` is not configured, the dashboard runs locally and saves feedback
to `data/feedback.csv` and policy updates to `models/nslkdd/q_table.json`. Cloud
hosts may discard those local changes after a restart.

## Included data

The test data and model files are committed for this demo. For public sharing,
review the repository contents before publishing. The bundled model is loaded
using Python serialization, so deploy only the files you trust.
