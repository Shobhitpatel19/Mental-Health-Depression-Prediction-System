import ImageGalleryWidget from "../ImageGalleryWidget";
import SearchBarHandle from "../SearchBarHandle";

import "./style.css";
import messages from "./messages.json";

function HomeContentWidget() {
    return (
        <div>
            <div className="central-content-container">
                <div className="hero-section">
                    <ImageGalleryWidget />
                    <div className="depression-prediction-system-info">
                        <p className="heading-text-block1">{messages["welcome_depression_prediction_system"]}</p>
                        <p className="journey-text-styles">{messages["together_lets_navigate_through_this_journey_toward"]}</p>
                    </div>
                </div>
            </div>

            <div className="search-bar">
                <SearchBarHandle />
            </div>
        </div>

    );
}

export default HomeContentWidget;
