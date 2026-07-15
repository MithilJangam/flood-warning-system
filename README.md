# flood-warning-system

An advanced, multi-model atmospheric modeling and risk prediction system featuring high-fidelity telemetry, interactive sandbox simulators, and intelligent analysis engines.

Developed by **Mithil Jangam**.

## Features
- **Disaster Dashboard**: Real-time telemetry reports and active weather station tracking.
- **Predictive Engine**: Localized disaster risk assessments computed using machine learning models (XGBoost, Decision Trees).
- **Intelligent Assistant**: Conversational safety intelligence chatbot using the Intelligent Analysis Engine.
- **Visual Risk Mapping**: Visual representation of geographical risks.

## Run Locally

**Prerequisites:**
- Node.js (v18 or higher recommended)
- Python (v3.x recommended)

1. **Configure Environment Variables:**
   Create a `.env` file in the root directory (refer to `.env.example`):
   ```properties
   AI_API_KEY="your-api-key"
   APP_URL="http://localhost:3000"
   OPENWEATHER_API_KEY="your-openweathermap-api-key"
   ```

2. **Run the Application (Recommended):**
   Run the Python launcher in your terminal. It will **automatically install all dependencies**, start the server, and launch the application in your browser:
   ```bash
   python app.py
   ```

   *(Alternative: If you do not want to use Python, you can manually run `npm install` and then `npm run dev` to launch the server.)*
