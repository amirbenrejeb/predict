# Flask MySQL Project

A basic Flask application with MySQL database integration.

## Setup

1. **Virtual Environment** (already done)
   ```bash
   python -m venv myenv
   myenv\Scripts\activate  # Windows
   ```

2. **Install Dependencies**
   ```bash
   pip install flask flask-sqlalchemy pymysql python-dotenv
   ```

3. **Database Configuration**
   - Update the `.env` file with your MySQL credentials
   - Make sure MySQL server is running
   - Create the database specified in `.env`

4. **Environment Variables**
   Update `.env` file with your actual values:
   ```
   DB_HOST=localhost
   DB_USER=your_mysql_username
   DB_PASSWORD=your_mysql_password
   DB_NAME=your_database_name
   SECRET_KEY=your-super-secret-key
   ```

5. **Run the Application**
   ```bash
   python app.py
   ```

## Project Structure

```
predict/
├── myenv/              # Virtual environment
├── app.py             # Main Flask application
├── models.py          # Database models
├── .env              # Environment variables (not in git)
└── README.md         # This file
```

## API Endpoints

- `GET /` - Basic hello world
- `GET /health` - Health check and database connection test

## Database Models

- **User**: Basic user model with username, email
- **Product**: Example product model (can be modified/removed)

## Next Steps

1. Configure your MySQL database
2. Update the `.env` file with your credentials
3. Run the application
4. Add more models, routes, and business logic as needed
