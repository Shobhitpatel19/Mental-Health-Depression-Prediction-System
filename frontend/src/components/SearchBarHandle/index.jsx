import React, { useState } from "react";
import axios from "axios";
import "./style.css";

function SearchBarHandle() {
  const [twitterHandle, setTwitterHandle] = useState('');
  const [predictionResult, setPredictionResult] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (event) => {
    setTwitterHandle(event.target.value);
  };


    const fetchData = async () => {
      if (!twitterHandle) return; // Don't make a request if twitterHandle is empty
      setLoading(true);
      setError('');

      try {
        const response = await axios.post('http://localhost:5000/tweets', {
            twitter_handle: twitterHandle
        }
      );

        console.log("Response: ",response.data)

        const predictionMessage = response.data.prediction === 1 ? 'User is depressed' : 'User is not depressed';
        setPredictionResult(predictionMessage);


      } catch (error) {
        console.error('Error fetching tweets:', error);
        setError('Error fetching tweets. Please try again.');
      } finally {
        setLoading(false);
      }
    };




  const handleSubmit = async (event) => {
    event.preventDefault();
    // No need to do anything here, useEffect will handle the API call
    fetchData();
  };

  return (
    <div className="search-container">
      <form onSubmit={handleSubmit}>
        <input
          className="search-input"
          type="text"
          placeholder="Enter Twitter Handle.."
          value={twitterHandle}
          onChange={handleChange}
        />
        <button onClick={handleSubmit} type="submit" className="search-button">
          <i className="fa fa-search"></i> Go
        </button>
      </form>

      {loading ? (<div>Loading...</div>) : error ? (<div>Error: {error}</div>) : (<div> {predictionResult}</div>)}
    </div>
  );
}

export default SearchBarHandle;
