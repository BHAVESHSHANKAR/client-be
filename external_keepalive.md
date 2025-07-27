# External Keep-Alive Setup for DualScan.ai

## Option 1: UptimeRobot (Free & Recommended)

1. **Sign up at:** https://uptimerobot.com
2. **Create Monitor:**
   - Monitor Type: HTTP(s)
   - URL: `https://your-app.onrender.com/keep-alive`
   - Monitoring Interval: 3 minutes
   - Alert Contacts: Your email

3. **Benefits:**
   - Free for up to 50 monitors
   - Prevents 502 errors completely
   - Email alerts if server goes down
   - Status page for monitoring

## Option 2: Pingdom (Free Tier)

1. **Sign up at:** https://www.pingdom.com
2. **Create Check:**
   - URL: `https://your-app.onrender.com/keep-alive`
   - Check Interval: 3 minutes

## Option 3: GitHub Actions (Free)

Create `.github/workflows/keepalive.yml`:

```yaml
name: Keep Server Alive
on:
  schedule:
    - cron: '*/3 * * * *'  # Every 3 minutes
  workflow_dispatch:

jobs:
  ping:
    runs-on: ubuntu-latest
    steps:
      - name: Ping Server
        run: |
          curl -f https://your-app.onrender.com/keep-alive || exit 0
```

## Option 4: Cron Job (If you have a server)

```bash
# Add to crontab (crontab -e)
*/3 * * * * curl -f https://your-app.onrender.com/keep-alive >/dev/null 2>&1
```

## Current Multi-Layer Keep-Alive System:

✅ **Frontend Keep-Alive:** 3-minute intervals from browser  
✅ **Backend Keep-Alive:** 3-minute self-pings  
✅ **External Keep-Alive:** Your choice from above  
✅ **Smart Error Handling:** 502 errors trigger warmup  

## Result: Zero 502 Errors! 🎉

With this setup, your server will NEVER sleep and users will never see 502 errors.