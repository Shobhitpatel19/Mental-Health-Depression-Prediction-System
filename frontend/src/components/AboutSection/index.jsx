import "./style.css";
import messages from "./messages.json";

function AboutSection() {
    return (
        <div>
            <div className="about-container">
                <img src="/images/aboutUs.png" className="about-image-container" alt=""/>
                <div className="info-card">
                    <p className="hero-text">{messages["about_us"]}</p>
                    <p className="mental-health-description-text-style">{messages["ltbr_gtwelcome_emoticheck_trusted_platform_depress"]}</p>
                </div>
            </div>

            <div className="mission-container">
                <p className="mission-statement-heading">{messages["mission"]}</p>
                <div className="mission-section">
                    <p className="mission-statement-text-style">{messages["mission_emoticheck_raise_awareness_about_depressio"]}</p>
                    <img src="/images/aboutpage.jpg" className="emoti-check-mission-illustration" alt="" />
                </div>
            </div>

            <div className="whatWeDo-content-container">
                <p className="WhatWeDo-text">{messages["what_we"]}</p>
                <p className="informative-text-style">{messages["depression_prediction_using_advanced_machine_learn"]}</p>
            </div>

        </div>

    );
}

export default AboutSection;