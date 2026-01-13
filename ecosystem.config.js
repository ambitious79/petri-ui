module.exports = {
  apps: [{
    name: 'petri-ui',
    script: 'app.py',
    interpreter: '/home/trishool/petri-ui/.venv/bin/python3',
    cwd: '/home/trishool/petri-ui',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G',
    env: {
      FLASK_ENV: 'production',
      PYTHONUNBUFFERED: '1'  // Important for real-time log output
    },
    error_file: './logs/pm2-error.log',
    out_file: './logs/pm2-out.log',
    log_file: './logs/pm2-combined.log',
    time: true,
    merge_logs: true,
    log_date_format: 'YYYY-MM-DD HH:mm:ss Z'
  }]
};

