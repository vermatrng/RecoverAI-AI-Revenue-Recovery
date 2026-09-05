
# Final Submission Checklist

## Demo
- [ ] Dashboard opens at `/app`
- [ ] API health is green
- [ ] Create Demo Payment works
- [ ] Analyze calls backend
- [ ] Recovery action changes transaction status
- [ ] Recovery Queue displays transactions
- [ ] Analytics page is ready

## ML
- [ ] Run `python ml/train_model.py`
- [ ] Confirm `ml/recovery_model.pkl` exists
- [ ] Confirm `data/transactions.csv` exists

## Deployment
- [ ] Push project to GitHub
- [ ] Deploy using Docker/Render or another host
- [ ] Set environment variables
- [ ] Test `/api/health`
- [ ] Test `/app`
- [ ] Configure Razorpay Test Mode webhook after deployment

## Security
- [ ] Never upload `.env`
- [ ] Never expose Razorpay key secret
- [ ] Use Test Mode for demonstration
