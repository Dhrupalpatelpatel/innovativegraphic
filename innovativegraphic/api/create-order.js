const axios = require('axios');

module.exports = async (req, res) => {
  try {
    const { name, email, amount, bundle } = req.body;

    const response = await axios.post(
      "https://sandbox.cashfree.com/pg/orders",
      {
        order_id: `order_${Date.now()}`,
        order_amount: amount,
        order_currency: "INR",
        customer_details: {
          customer_id: email,
          customer_name: name,
          customer_email: email,
          customer_phone: "9999999999"
        },
        order_meta: {
          return_url: "https://innovativegraphic.vercel.app/thank-you?order_id={order_id}",
        }
      },
      {
        headers: {
          "x-client-id": process.env.CASHFREE_APP_ID, // Make sure this matches Vercel env var name
          "x-client-secret": process.env.CASHFREE_SECRET_KEY, // Make sure this matches
          "x-api-version": "2022-09-01"
        }
      }
    );

    res.json({
      paymentSessionId: response.data.payment_session_id
    });
    
  } catch (error) {
    console.error("Cashfree API error:", error.response?.data || error.message);
    res.status(500).json({ 
      error: "Payment failed. Please try again.",
      details: error.message
    });
  }
};
