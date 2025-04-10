const axios = require('axios');

module.exports = async (req, res) => {
  // Log the incoming request body to see what the frontend is sending
  // In production, be cautious about logging sensitive data.
  console.log("INFO: Received request to create order with body:", JSON.stringify(req.body, null, 2));

  try {
    const { name, email, amount, bundle } = req.body;

    // --- Basic Input Validation ---
    if (!name || !email || !amount || !bundle) {
      console.error("ERROR: Validation Failed - Missing required fields in request body.");
      // Send a clear error back to the frontend
      return res.status(400).json({
        error: "Missing required fields: name, email, amount, or bundle.",
      });
    }
    // Ensure amount is a positive number
    const numericAmount = parseInt(amount);
    if (isNaN(numericAmount) || numericAmount <= 0) {
       console.error("ERROR: Validation Failed - Invalid amount received:", amount);
       return res.status(400).json({
         error: "Invalid order amount provided.",
       });
    }
    // --- End Input Validation ---

    // --- Prepare Cashfree Request Data ---
    const orderPayload = {
      order_id: `order_${Date.now()}`, // Generate a unique order ID
      order_amount: numericAmount,     // Use the validated numeric amount
      order_currency: "INR",
      customer_details: {
        customer_id: email, // Using email as a unique customer identifier
        customer_name: name,
        customer_email: email,
        // NOTE: Hardcoded phone number is generally okay for sandbox testing,
        // but you might need a real or dynamic one for production.
        customer_phone: "9999999999"
      },
      order_meta: {
        // The URL Cashfree redirects the user to after payment attempt
        return_url: `https://innovativegraphic.vercel.app/thank-you?order_id={order_id}`,
        // Optional: You can set a notification URL for webhooks here too,
        // but it's often better managed in the Cashfree dashboard.
        // notify_url: "https://innovativegraphic.vercel.app/api/webhook"
      },
       // Optional: Add a note for your reference in the Cashfree dashboard
       order_note: `Order for bundle: ${bundle}`
    };

    // Prepare headers for Cashfree API call
    const cashfreeHeaders = {
      "Content-Type": "application/json",
      // Ensure these environment variable names EXACTLY match your Vercel setup
      "x-client-id": process.env.CASHFREE_APP_ID,
      "x-client-secret": process.env.CASHFREE_SECRET_KEY,
      "x-api-version": "2022-09-01", // Use a recent, valid API version
      // Optional but recommended: Helps Cashfree support trace specific requests if needed
      "x-request-id": `innovativegraphic-req-${Date.now()}`
    };

    // Define the Cashfree API endpoint (Sandbox)
    const cashfreeApiUrl = "https://sandbox.cashfree.com/pg/orders";

    // Log the data being sent TO Cashfree for debugging.
    // Mask the secret key in logs for security.
    console.log("INFO: Sending request to Cashfree URL:", cashfreeApiUrl);
    console.log("INFO: Cashfree Headers (Secret Masked):", JSON.stringify({ ...cashfreeHeaders, "x-client-secret": '*** MASKED ***' }, null, 2));
    console.log("INFO: Cashfree Payload:", JSON.stringify(orderPayload, null, 2));
    // --- End Prepare Cashfree Request Data ---


    // --- Make the API Call to Cashfree ---
    const response = await axios.post(cashfreeApiUrl, orderPayload, { headers: cashfreeHeaders });
    // --- End API Call ---

    // Log the successful response received FROM Cashfree
    console.log("INFO: Cashfree API Success Response:", JSON.stringify(response.data, null, 2));

    // --- Process Cashfree Response & Respond to Frontend ---
    // Check if the expected payment_session_id is present in Cashfree's response
    if (!response.data || !response.data.payment_session_id) {
        console.error("ERROR: Cashfree response data is missing 'payment_session_id'. Response:", JSON.stringify(response.data, null, 2));
        // Throw an error to be caught by the catch block below
        throw new Error("Failed to retrieve payment session ID from Cashfree.");
    }

    // Send the essential paymentSessionId back to your frontend
    res.status(200).json({
      paymentSessionId: response.data.payment_session_id
    });
    console.log("INFO: Successfully sent paymentSessionId to frontend.");
    // --- End Process Cashfree Response & Respond to Frontend ---

  } catch (error) {
    // --- Enhanced Error Handling ---
    console.error("💥💥 ERROR DURING CASHFREE ORDER CREATION 💥💥");

    // Check if the error is from the Axios request (Cashfree API response)
    if (error.response) {
      // The request was made and the server responded with a non-2xx status code
      console.error("ERROR: Axios Response Status:", error.response.status);
      console.error("ERROR: Axios Response Headers:", JSON.stringify(error.response.headers, null, 2));
      // ** THIS IS CRITICAL -> Log the actual error data from Cashfree **
      console.error("ERROR: Axios Response Data:", JSON.stringify(error.response.data, null, 2));

      // Construct a more informative error message for the frontend
      const errorMessage = error.response.data?.message || "Payment gateway rejected the request.";
      const errorType = error.response.data?.type; // e.g., 'authentication_error', 'invalid_request_error'
      const errorCode = error.response.data?.code; // Specific error code from Cashfree

      res.status(error.response.status || 500).json({
         error: `Payment failed: ${errorMessage}`,
         // Optionally include more details (be careful in production)
         details: {
             type: errorType,
             code: errorCode,
             message: errorMessage
         }
       });

    } else if (error.request) {
      // The request was made but no response was received (network issue, timeout)
      console.error("ERROR: No response received from Cashfree. Request details:", error.request);
      res.status(504).json({ // 504 Gateway Timeout might be appropriate
        error: "Payment gateway did not respond. Please check network or try again later.",
        details: "No response received from Cashfree API."
      });
    } else {
      // Something else went wrong setting up the request or in our code
      console.error("ERROR: Request setup or internal error:", error.message);
      console.error("ERROR: Stacktrace:", error.stack); // Log stack trace for internal errors
      res.status(500).json({
        error: "An internal server error occurred while setting up the payment request.",
        details: error.message
      });
    }
     // Log the configuration used (excluding secret) when an error occurs
     console.error("INFO: Environment Config Used (App ID):", process.env.CASHFREE_APP_ID);
     console.error("INFO: Cashfree URL Used:", "https://sandbox.cashfree.com/pg/orders");
    // --- End Enhanced Error Handling ---
  }
};
